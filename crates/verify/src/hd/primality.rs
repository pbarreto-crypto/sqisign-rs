//! The goodness test on the response degree `q`, as the verifier repeats it:
//! `2^e − q` must be a probable prime congruent to 1 modulo 4 (the paper's
//! Section 4, "The degree"). The signer runs 40 Miller–Rabin rounds on random
//! bases; the verifier, which must be deterministic, runs 40 rounds on the
//! first 40 primes. Variable time; `q` is public data. Generic over the width
//! of the integers ([`Uint`]), with Montgomery arithmetic from [`Monty`].

use super::uint::{Monty, Uint};

/// Number of Miller–Rabin rounds, as in the signer.
pub const MR_ROUNDS: usize = 40;

/// The first [`MR_ROUNDS`] primes, the verifier's fixed bases.
const BASES: [u64; MR_ROUNDS] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
];

/// Miller–Rabin on the fixed bases of [`BASES`]. `false` for `n < 2` and for
/// even `n > 2`.
pub fn is_probable_prime<const N: usize>(n: &Uint<N>) -> bool {
    if n.lt(&Uint::from_u64(2)) {
        return false;
    }
    if !n.is_odd() {
        return *n == Uint::from_u64(2);
    }
    // Small n: trial division by the bases decides it.
    for &p in BASES.iter() {
        if *n == Uint::from_u64(p) {
            return true;
        }
        if n.rem_u64(p) == 0 {
            return false;
        }
    }
    let mont = Monty::new(n);
    let one = mont.to(&Uint::ONE);
    let minus_one = n.wrapping_sub(&one); // n − (R mod n) = −1 in Montgomery form
    let nm1 = n.wrapping_sub(&Uint::ONE);
    let mut s = 0u32;
    while (nm1.0[(s / 64) as usize] >> (s % 64)) & 1 == 0 {
        s += 1;
    }
    let d = nm1.shr(s);
    'bases: for &p in BASES.iter() {
        let mut x = mont.pow(&mont.to(&Uint::from_u64(p)), &d);
        if x == one || x == minus_one {
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
/// (mod 4)` and `2^e − q` a probable prime ([`is_probable_prime`]). `false`
/// if `2^e` does not fit the width.
pub fn is_good_degree<const N: usize>(e: u32, q: &Uint<N>) -> bool {
    if q.bits() > e || e >= Uint::<N>::BITS {
        return false;
    }
    let n = Uint::<N>::pow2(e).wrapping_sub(q);
    n.0[0] & 3 == 1 && is_probable_prime(&n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hd::uint::{U3, U4};

    #[test]
    fn known_primes_and_composites() {
        // 2^127 − 1 and 2^89 − 1 are prime; their product and 2^128 + 1 are not.
        let m127 = U4::pow2(127).wrapping_sub(&U4::ONE);
        let m89 = U4::pow2(89).wrapping_sub(&U4::ONE);
        assert!(is_probable_prime(&m127));
        assert!(is_probable_prime(&m89));
        assert!(!is_probable_prime(&m127.wrapping_mul(&m89)));
        assert!(!is_probable_prime(&U4::pow2(128).wrapping_add(&U4::ONE)));
        for p in [2u64, 3, 5, 173, 179, 65537] {
            assert!(is_probable_prime(&U3::from_u64(p)), "{p}");
        }
        for c in [1u64, 4, 9, 341, 561, 65535] {
            assert!(!is_probable_prime(&U3::from_u64(c)), "{c}");
        }
    }

    #[test]
    fn widths_agree() {
        // The 192-bit and 256-bit versions agree on Miller–Rabin and on the
        // goodness test, for degrees of the level-I size.
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut good = 0;
        for _ in 0..400 {
            let limbs = [next(), next(), next() & ((1 << 46) - 1)];
            let q3 = Uint::<3>(limbs);
            let q4 = Uint::<4>([limbs[0], limbs[1], limbs[2], 0]);
            let (a, b) = (is_good_degree(174, &q3), is_good_degree(174, &q4));
            assert_eq!(a, b);
            good += a as usize;
            let n3 = Uint::<3>([limbs[0] | 1, limbs[1], limbs[2]]);
            let n4 = Uint::<4>([limbs[0] | 1, limbs[1], limbs[2], 0]);
            assert_eq!(is_probable_prime(&n3), is_probable_prime(&n4));
        }
        assert!(good > 0, "some random degrees are good");
    }
}
