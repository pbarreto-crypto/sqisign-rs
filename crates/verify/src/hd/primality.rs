//! The goodness test on the response degree `q`, as the verifier repeats it:
//! `2^e − q` must be a probable prime congruent to 1 modulo 4 (the paper's
//! Section 4, "The degree"). The signer runs 40 Miller–Rabin rounds on random
//! bases; the verifier, which must be deterministic, runs 40 rounds on the
//! first 40 primes. Variable time; `q` is public data.
//!
//! Exponentiation uses Montgomery multiplication on four 64-bit limbs, for an
//! odd modulus below `2^255`; the bit-serial [`U4::mul_mod`] would make the 40
//! rounds cost as much as the rest of the verification.

use super::uint::U4;

/// Number of Miller–Rabin rounds, as in the signer.
pub const MR_ROUNDS: usize = 40;

/// The first [`MR_ROUNDS`] primes, the verifier's fixed bases.
const BASES: [u64; MR_ROUNDS] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
];

/// Montgomery arithmetic modulo an odd `n < 2^255`, with `R = 2^256`.
struct Mont {
    n: U4,
    /// `−n^{-1} mod 2^64`.
    n_inv: u64,
    /// `R mod n` (one in Montgomery form).
    one: U4,
    /// `R^2 mod n`.
    r2: U4,
}

impl Mont {
    fn new(n: &U4) -> Option<Mont> {
        if !n.is_odd() || n.bits() > 255 {
            return None;
        }
        // Newton iteration for n[0]^{-1} mod 2^64 (each step doubles the bits).
        let n0 = n.0[0];
        let mut inv: u64 = 1;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(inv)));
        }
        // 2^k mod n by doubling; 2x < 2^256 since x < n < 2^255.
        let mut x = U4::ONE.div_rem(n).1;
        let mut one = U4::ZERO;
        for k in 1..=512 {
            x = x.shl(1);
            if !x.lt(n) {
                x = x.wrapping_sub(n);
            }
            if k == 256 {
                one = x;
            }
        }
        Some(Mont { n: *n, n_inv: inv.wrapping_neg(), one, r2: x })
    }

    /// `a b R^{-1} mod n` for `a, b < n` (CIOS).
    fn mul(&self, a: &U4, b: &U4) -> U4 {
        let n = &self.n.0;
        let mut t = [0u64; 6];
        for i in 0..4 {
            let mut carry: u128 = 0;
            for j in 0..4 {
                let s = t[j] as u128 + (a.0[j] as u128) * (b.0[i] as u128) + carry;
                t[j] = s as u64;
                carry = s >> 64;
            }
            let s = t[4] as u128 + carry;
            t[4] = s as u64;
            t[5] = (s >> 64) as u64;

            let m = t[0].wrapping_mul(self.n_inv);
            let s = t[0] as u128 + (m as u128) * (n[0] as u128);
            let mut carry = s >> 64;
            for j in 1..4 {
                let s = t[j] as u128 + (m as u128) * (n[j] as u128) + carry;
                t[j - 1] = s as u64;
                carry = s >> 64;
            }
            let s = t[4] as u128 + carry;
            t[3] = s as u64;
            t[4] = t[5] + (s >> 64) as u64;
        }
        let r = U4([t[0], t[1], t[2], t[3]]);
        if t[4] != 0 || !r.lt(&self.n) {
            r.wrapping_sub(&self.n)
        } else {
            r
        }
    }

    fn to_mont(&self, a: &U4) -> U4 {
        self.mul(&a.div_rem(&self.n).1, &self.r2)
    }

    /// `base^e` for `base` in Montgomery form; the result in Montgomery form.
    fn pow(&self, base: &U4, e: &U4) -> U4 {
        let mut r = self.one;
        for i in (0..e.bits()).rev() {
            r = self.mul(&r, &r);
            if (e.0[(i / 64) as usize] >> (i % 64)) & 1 == 1 {
                r = self.mul(&r, base);
            }
        }
        r
    }
}

/// Miller–Rabin on the fixed bases of [`BASES`]. `false` for `n < 2`, for
/// even `n > 2`, and for `n ≥ 2^255`, beyond the compact format, which is built
/// at level I only (`e = 174`).
pub fn is_probable_prime(n: &U4) -> bool {
    if n.lt(&U4::from_u64(2)) {
        return false;
    }
    if !n.is_odd() {
        return *n == U4::from_u64(2);
    }
    // Small n: trial division by the bases decides it.
    for &p in BASES.iter() {
        if *n == U4::from_u64(p) {
            return true;
        }
        if n.div_rem(&U4::from_u64(p)).1.is_zero() {
            return false;
        }
    }
    let mont = match Mont::new(n) {
        Some(m) => m,
        None => return false,
    };
    let nm1 = n.wrapping_sub(&U4::ONE);
    let mut s = 0u32;
    while (nm1.0[(s / 64) as usize] >> (s % 64)) & 1 == 0 {
        s += 1;
    }
    let d = nm1.shr(s);
    let minus_one = n.wrapping_sub(&mont.one); // n − R mod n = −1 in Montgomery form
    'bases: for &p in BASES.iter() {
        let mut x = mont.pow(&mont.to_mont(&U4::from_u64(p)), &d);
        if x == mont.one || x == minus_one {
            continue;
        }
        for _ in 1..s {
            x = mont.mul(&x, &x);
            if x == minus_one {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// The verifier's goodness test on the degree: `q < 2^e`, `2^e − q ≡ 1
/// (mod 4)` and `2^e − q` a probable prime ([`is_probable_prime`]).
pub fn is_good_degree(e: u32, q: &U4) -> bool {
    if q.bits() > e || e > 255 {
        return false;
    }
    let n = U4::pow2(e).wrapping_sub(q);
    n.0[0] & 3 == 1 && is_probable_prime(&n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(x: u128) -> U4 {
        U4([x as u64, (x >> 64) as u64, 0, 0])
    }

    #[test]
    fn montgomery_matches_bit_serial() {
        // Odd moduli of various sizes against U4::pow_mod.
        let mods = [
            u(1_000_000_007),
            u((1u128 << 89) - 1),
            U4([0xffff_ffff_ffff_ffc5, 0xffff_ffff_ffff_ffff, 0x3fff, 0]),
            U4([0x1234_5678_9abc_def1, 0x0fed_cba9_8765_4321, 0xdead_beef_cafe_babe, 0x7fff_0000_1111_2223]),
        ];
        let mut seed = 0x5eedu64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };
        for n in mods.iter() {
            let m = Mont::new(n).expect("odd modulus below 2^255");
            for _ in 0..20 {
                let a = U4([next(), next(), next(), next() >> 2]).div_rem(n).1;
                let e = U4([next(), next(), next() & 0x3fff, 0]);
                let slow = a.pow_mod(&e, n);
                let fast = m.mul(&m.pow(&m.to_mont(&a), &e), &U4::ONE); // out of Montgomery form
                assert_eq!(fast, slow);
            }
        }
    }

    #[test]
    fn known_primes_and_composites() {
        for p in [2u128, 3, 5, 97, 1_000_000_007, (1u128 << 89) - 1, (1u128 << 127) - 1] {
            assert!(is_probable_prime(&u(p)), "{} is prime", p);
        }
        // Carmichael numbers, strong pseudoprimes to small bases, a square of
        // a prime, and a product of two large primes.
        let composites = [
            561u128,
            41041,
            2047,                                     // strong pseudoprime to base 2
            3_215_031_751,                            // to bases 2, 3, 5, 7
            3_825_123_056_546_413_051,                // to the primes up to 23
            ((1u128 << 61) - 1) * ((1u128 << 61) - 1),
            ((1u128 << 61) - 1) * 1_000_000_007,
        ];
        for c in composites {
            assert!(!is_probable_prime(&u(c)), "{} is composite", c);
        }
    }

    #[test]
    fn degree_goodness() {
        // 2^e − q with e = 5: q = 3 gives 29 (prime, ≡ 1 mod 4), good;
        // q = 7 gives 25 (≡ 1 mod 4, a square), not good; q = 1 gives 31
        // (prime, ≡ 3 mod 4), not good; q ≥ 2^e is not good.
        assert!(is_good_degree(5, &u(3)));
        assert!(!is_good_degree(5, &u(7)));
        assert!(!is_good_degree(5, &u(1)));
        assert!(!is_good_degree(5, &u(40)));
    }
}
