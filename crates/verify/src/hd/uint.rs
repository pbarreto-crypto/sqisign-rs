//! Fixed-width unsigned integers on `N` little-endian `u64` limbs, for the
//! norm equation `2^e − q = a_1^2 + a_2^2` of the compact format: the response
//! degree `q`, Cornacchia's descent, the square root of `−1` modulo a prime by
//! exponentiation, and the goodness test. Each level uses the width it needs
//! (`U3`, 192 bits, at level I, where `e = 174`). Variable time; everything
//! here is public data (the signature's `q`). The 128-bit products below are
//! the usual double-width products of two limbs.

/// The largest width supported, in limbs (512 bits); scratch buffers are
/// sized from it.
const MAX_LIMBS: usize = 8;

/// An unsigned integer on `N` little-endian `u64` limbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uint<const N: usize>(pub [u64; N]);

/// 192 bits: the response degree at level I (`e = 174`).
pub type U3 = Uint<3>;
/// 256 bits.
pub type U4 = Uint<4>;

impl<const N: usize> Default for Uint<N> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const N: usize> Uint<N> {
    pub const ZERO: Self = Uint([0; N]);
    pub const ONE: Self = {
        let mut a = [0u64; N];
        a[0] = 1;
        Uint(a)
    };
    pub const BITS: u32 = 64 * N as u32;

    pub fn from_u64(x: u64) -> Self {
        let mut r = Self::ZERO;
        r.0[0] = x;
        r
    }

    /// `2^k`, for `k < 64 N`.
    pub fn pow2(k: u32) -> Self {
        let mut r = Self::ZERO;
        r.0[(k / 64) as usize] = 1u64 << (k % 64);
        r
    }

    /// From little-endian bytes, at most `8 N` of them.
    pub fn from_le_bytes(b: &[u8]) -> Self {
        let mut r = Self::ZERO;
        for (i, byte) in b.iter().enumerate().take(8 * N) {
            r.0[i / 8] |= (*byte as u64) << (8 * (i % 8));
        }
        r
    }

    /// Writes the low `out.len()` bytes, little-endian (`out.len() ≤ 8 N`).
    pub fn write_le_bytes(&self, out: &mut [u8]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = (self.0[i / 8] >> (8 * (i % 8))) as u8;
        }
    }

    pub fn low_u128(&self) -> u128 {
        let hi = if N > 1 { self.0[1] } else { 0 };
        (self.0[0] as u128) | ((hi as u128) << 64)
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }

    pub fn is_odd(&self) -> bool {
        self.0[0] & 1 == 1
    }

    pub fn bits(&self) -> u32 {
        for i in (0..N).rev() {
            if self.0[i] != 0 {
                return 64 * i as u32 + 64 - self.0[i].leading_zeros();
            }
        }
        0
    }

    pub fn compare(&self, other: &Self) -> core::cmp::Ordering {
        for i in (0..N).rev() {
            match self.0[i].cmp(&other.0[i]) {
                core::cmp::Ordering::Equal => continue,
                o => return o,
            }
        }
        core::cmp::Ordering::Equal
    }

    pub fn lt(&self, other: &Self) -> bool {
        self.compare(other) == core::cmp::Ordering::Less
    }

    pub fn wrapping_add(&self, other: &Self) -> Self {
        let mut r = Self::ZERO;
        let mut carry = 0u128;
        for i in 0..N {
            let s = self.0[i] as u128 + other.0[i] as u128 + carry;
            r.0[i] = s as u64;
            carry = s >> 64;
        }
        r
    }

    pub fn wrapping_sub(&self, other: &Self) -> Self {
        let mut r = Self::ZERO;
        let mut borrow = 0u64;
        for i in 0..N {
            let (d1, b1) = self.0[i].overflowing_sub(other.0[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            r.0[i] = d2;
            borrow = (b1 | b2) as u64;
        }
        r
    }

    pub fn wrapping_mul(&self, other: &Self) -> Self {
        let w = self.mul_wide(other);
        let mut r = Self::ZERO;
        r.0.copy_from_slice(&w[..N]);
        r
    }

    /// The full product; its first `2 N` limbs are significant.
    pub fn mul_wide(&self, other: &Self) -> [u64; 2 * MAX_LIMBS] {
        let mut w = [0u64; 2 * MAX_LIMBS];
        for i in 0..N {
            let mut carry = 0u128;
            for j in 0..N {
                let acc = w[i + j] as u128 + (self.0[i] as u128) * (other.0[j] as u128) + carry;
                w[i + j] = acc as u64;
                carry = acc >> 64;
            }
            w[i + N] = carry as u64;
        }
        w
    }

    pub fn shl(&self, k: u32) -> Self {
        let mut r = Self::ZERO;
        let limbs = (k / 64) as usize;
        let bits = k % 64;
        for i in (limbs..N).rev() {
            let src = i - limbs;
            r.0[i] = self.0[src] << bits;
            if bits != 0 && src > 0 {
                r.0[i] |= self.0[src - 1] >> (64 - bits);
            }
        }
        r
    }

    pub fn shr(&self, k: u32) -> Self {
        let mut r = Self::ZERO;
        let limbs = (k / 64) as usize;
        let bits = k % 64;
        for i in 0..N.saturating_sub(limbs) {
            let src = i + limbs;
            r.0[i] = self.0[src] >> bits;
            if bits != 0 && src + 1 < N {
                r.0[i] |= self.0[src + 1] << (64 - bits);
            }
        }
        r
    }

    #[inline]
    fn limbs(&self) -> usize {
        (0..N).rev().find(|&i| self.0[i] != 0).map_or(0, |i| i + 1)
    }

    pub fn rem_u64(&self, d: u64) -> u64 {
        let dv = d as u128;
        let mut rem = 0u128;
        for i in (0..N).rev() {
            rem = (((rem << 64) | self.0[i] as u128) % dv) & u64::MAX as u128;
        }
        rem as u64
    }

    /// Quotient and remainder (Knuth's Algorithm D).
    pub fn div_rem(&self, d: &Self) -> (Self, Self) {
        debug_assert!(!d.is_zero());
        if self.lt(d) {
            return (Self::ZERO, *self);
        }
        let n = d.limbs();
        if n == 1 {
            let dv = d.0[0] as u128;
            let mut q = Self::ZERO;
            let mut rem = 0u128;
            for i in (0..N).rev() {
                let cur = (rem << 64) | self.0[i] as u128;
                q.0[i] = (cur / dv) as u64;
                rem = cur % dv;
            }
            return (q, Self::from_u64(rem as u64));
        }
        // Normalise so that the divisor's top limb has its top bit set.
        let sh = d.0[n - 1].leading_zeros();
        let mut un = [0u64; MAX_LIMBS + 1];
        let mut vn = [0u64; MAX_LIMBS];
        if sh == 0 {
            un[..N].copy_from_slice(&self.0);
            vn[..N].copy_from_slice(&d.0);
        } else {
            for i in (1..N).rev() {
                un[i] = (self.0[i] << sh) | (self.0[i - 1] >> (64 - sh));
                vn[i] = (d.0[i] << sh) | (d.0[i - 1] >> (64 - sh));
            }
            un[N] = self.0[N - 1] >> (64 - sh);
            un[0] = self.0[0] << sh;
            vn[0] = d.0[0] << sh;
        }
        let m = N - n;
        let mut q = Self::ZERO;
        let b: u128 = 1 << 64;
        for j in (0..=m).rev() {
            let num = ((un[j + n] as u128) << 64) | un[j + n - 1] as u128;
            let mut qhat = num / vn[n - 1] as u128;
            let mut rhat = num % vn[n - 1] as u128;
            while qhat >= b || qhat * vn[n - 2] as u128 > ((rhat << 64) | un[j + n - 2] as u128) {
                qhat -= 1;
                rhat += vn[n - 1] as u128;
                if rhat >= b {
                    break;
                }
            }
            // Multiply and subtract.
            let mut borrow: i128 = 0;
            let mut carry: u128 = 0;
            for i in 0..n {
                let pr = qhat * vn[i] as u128 + carry;
                carry = pr >> 64;
                let t = un[i + j] as i128 - borrow - (pr as u64) as i128;
                un[i + j] = t as u64;
                borrow = if t < 0 { 1 } else { 0 };
            }
            let t = un[j + n] as i128 - borrow - carry as i128;
            un[j + n] = t as u64;
            if t < 0 {
                // The estimate was one too large: add the divisor back.
                qhat -= 1;
                let mut c: u128 = 0;
                for i in 0..n {
                    let t = un[i + j] as u128 + vn[i] as u128 + c;
                    un[i + j] = t as u64;
                    c = t >> 64;
                }
                un[j + n] = un[j + n].wrapping_add(c as u64);
            }
            q.0[j] = qhat as u64;
        }
        // The remainder is un[0..n], shifted back.
        let mut r = Self::ZERO;
        if sh == 0 {
            r.0[..n].copy_from_slice(&un[..n]);
        } else {
            for i in 0..n {
                r.0[i] = (un[i] >> sh) | (un[i + 1] << (64 - sh));
            }
        }
        (q, r)
    }

    pub fn mul_mod(&self, other: &Self, m: &Self) -> Self {
        let mt = Monty::new(m);
        mt.from(&mt.mul(&mt.to(self), &mt.to(other)))
    }

    pub fn pow_mod(&self, e: &Self, m: &Self) -> Self {
        let mt = Monty::new(m);
        mt.from(&mt.pow(&mt.to(self), e))
    }

    /// `⌊√self⌋` by Newton's iteration.
    pub fn isqrt(&self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }
        let mut x = Self::ONE.shl(self.bits().div_ceil(2));
        loop {
            let (q, _) = self.div_rem(&x);
            let x_new = x.wrapping_add(&q).shr(1);
            if !x_new.lt(&x) {
                return x;
            }
            x = x_new;
        }
    }
}

impl<const N: usize> Uint<N> {
    /// The low 64 bits.
    #[inline]
    pub fn low_u64(&self) -> u64 {
        self.0[0]
    }

    /// From little-endian limbs, the first `N` of them (missing ones are zero).
    pub fn from_le_limbs(l: &[u64]) -> Self {
        let mut r = Self::ZERO;
        for (d, s) in r.0.iter_mut().zip(l.iter()) {
            *d = *s;
        }
        r
    }

    /// The same value on `M` limbs, truncated or zero-extended.
    pub fn resize<const M: usize>(&self) -> Uint<M> {
        Uint::<M>::from_le_limbs(&self.0)
    }

    /// `2^k − 1`, for `k ≤ 64 N`.
    pub fn mask(k: u32) -> Self {
        if k >= Self::BITS {
            return Self([u64::MAX; N]);
        }
        Self::pow2(k).wrapping_sub(&Self::ONE)
    }

    /// `−self` modulo `2^(64 N)`.
    #[inline]
    pub fn wrapping_neg(&self) -> Self {
        Self::ZERO.wrapping_sub(self)
    }

    /// Bit `k`.
    #[inline]
    pub fn bit(&self, k: u32) -> bool {
        (self.0[(k / 64) as usize] >> (k % 64)) & 1 == 1
    }

    /// The number of trailing zero bits (`64 N` for zero).
    pub fn trailing_zeros(&self) -> u32 {
        for i in 0..N {
            if self.0[i] != 0 {
                return 64 * i as u32 + self.0[i].trailing_zeros();
            }
        }
        Self::BITS
    }

    /// The inverse of an odd `self` modulo `2^k` (`k ≤ 64 N`), by Newton's
    /// iteration: each step doubles the number of correct low bits, so the
    /// number of steps follows `k`.
    pub fn inv_mod_pow2(&self, k: u32) -> Self {
        let m = Self::mask(k);
        let a = *self & m;
        let mut inv = Self::ONE;
        let mut good = 1u32;
        while good < k {
            inv = (inv * (Self::from_u64(2) - a * inv)) & m;
            good *= 2;
        }
        inv & m
    }
}

/// Wrapping arithmetic modulo `2^(64 N)` and bitwise operations, by value, so
/// that scalars modulo `2^r` read as they would with primitive integers.
impl<const N: usize> core::ops::Add for Uint<N> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        self.wrapping_add(&o)
    }
}
impl<const N: usize> core::ops::Sub for Uint<N> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        self.wrapping_sub(&o)
    }
}
impl<const N: usize> core::ops::Mul for Uint<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        self.wrapping_mul(&o)
    }
}
impl<const N: usize> core::ops::Neg for Uint<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}
impl<const N: usize> core::ops::BitAnd for Uint<N> {
    type Output = Self;
    #[inline]
    fn bitand(self, o: Self) -> Self {
        Uint(core::array::from_fn(|i| self.0[i] & o.0[i]))
    }
}
impl<const N: usize> core::ops::BitOr for Uint<N> {
    type Output = Self;
    #[inline]
    fn bitor(self, o: Self) -> Self {
        Uint(core::array::from_fn(|i| self.0[i] | o.0[i]))
    }
}
impl<const N: usize> core::ops::BitXor for Uint<N> {
    type Output = Self;
    #[inline]
    fn bitxor(self, o: Self) -> Self {
        Uint(core::array::from_fn(|i| self.0[i] ^ o.0[i]))
    }
}
impl<const N: usize> core::ops::Shl<u32> for Uint<N> {
    type Output = Self;
    #[inline]
    fn shl(self, k: u32) -> Self {
        Uint::shl(&self, k)
    }
}
impl<const N: usize> core::ops::Shr<u32> for Uint<N> {
    type Output = Self;
    #[inline]
    fn shr(self, k: u32) -> Self {
        Uint::shr(&self, k)
    }
}
impl<const N: usize> PartialOrd for Uint<N> {
    #[inline]
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> {
        Some(self.compare(o))
    }
}
impl<const N: usize> Ord for Uint<N> {
    #[inline]
    fn cmp(&self, o: &Self) -> core::cmp::Ordering {
        self.compare(o)
    }
}
impl<const N: usize> From<u64> for Uint<N> {
    #[inline]
    fn from(x: u64) -> Self {
        Self::from_u64(x)
    }
}

impl Uint<4> {
    /// The 32 little-endian bytes.
    pub fn to_le_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        self.write_le_bytes(&mut out);
        out
    }
}

impl Uint<3> {
    /// The 24 little-endian bytes.
    pub fn to_le_bytes(&self) -> [u8; 24] {
        let mut out = [0u8; 24];
        self.write_le_bytes(&mut out);
        out
    }
}

/// Montgomery arithmetic modulo an odd `n` of `N` limbs, `R = 2^(64 N)`.
pub struct Monty<const N: usize> {
    n: Uint<N>,
    n0: u64,
    r2: Uint<N>,
}

impl<const N: usize> Monty<N> {
    pub fn new(n: &Uint<N>) -> Self {
        debug_assert!(n.is_odd());
        // n^-1 mod 2^64 by Newton's iteration from the 3-bit seed n itself.
        let mut inv = n.0[0];
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n.0[0].wrapping_mul(inv)));
        }
        // R mod n = (2^(64 N) - n) mod n, then R^2 mod n by 64 N doublings.
        let mut r = Uint::<N>::ZERO.wrapping_sub(n).div_rem(n).1;
        for _ in 0..64 * N {
            let (s, c) = (r.wrapping_add(&r), r.0[N - 1] >> 63);
            r = if c != 0 || !s.lt(n) { s.wrapping_sub(n) } else { s };
        }
        Monty {
            n: *n,
            n0: inv.wrapping_neg(),
            r2: r,
        }
    }

    /// Montgomery product `a b R^-1 mod n` (CIOS). The accumulator is `N`
    /// limbs plus two carry words held apart, so that it can stay in
    /// registers whatever `N` is.
    #[allow(clippy::needless_range_loop)] // the limb index runs over three arrays at once
    #[inline(always)]
    pub fn mul(&self, a: &Uint<N>, b: &Uint<N>) -> Uint<N> {
        let n = &self.n.0;
        let mut t = [0u64; N];
        let mut tn: u64 = 0;
        for i in 0..N {
            let bi = b.0[i] as u128;
            let mut c: u128 = 0;
            for j in 0..N {
                let s = t[j] as u128 + a.0[j] as u128 * bi + c;
                t[j] = s as u64;
                c = s >> 64;
            }
            let s = tn as u128 + c;
            tn = s as u64;
            let tn1 = (s >> 64) as u64;
            let m = (t[0].wrapping_mul(self.n0)) as u128;
            let s = t[0] as u128 + m * n[0] as u128;
            let mut c = s >> 64;
            for j in 1..N {
                let s = t[j] as u128 + m * n[j] as u128 + c;
                t[j - 1] = s as u64;
                c = s >> 64;
            }
            let s = tn as u128 + c;
            t[N - 1] = s as u64;
            tn = tn1 + (s >> 64) as u64;
        }
        let r = Uint(t);
        if tn != 0 || !r.lt(&self.n) {
            r.wrapping_sub(&self.n)
        } else {
            r
        }
    }

    pub fn to(&self, x: &Uint<N>) -> Uint<N> {
        let x = if x.lt(&self.n) {
            *x
        } else {
            x.div_rem(&self.n).1
        };
        self.mul(&x, &self.r2)
    }

    pub fn from(&self, x: &Uint<N>) -> Uint<N> {
        self.mul(x, &Uint::<N>::ONE)
    }

    pub fn pow(&self, base: &Uint<N>, e: &Uint<N>) -> Uint<N> {
        let one = self.to(&Uint::<N>::ONE);
        let mut table = [one; 16];
        for i in 1..16 {
            table[i] = self.mul(&table[i - 1], base);
        }
        let bits = e.bits();
        if bits == 0 {
            return one;
        }
        let windows = bits.div_ceil(4);
        let mut acc = one;
        for w in (0..windows).rev() {
            if w + 1 < windows {
                for _ in 0..4 {
                    acc = self.mul(&acc, &acc);
                }
            }
            let shift = 4 * w;
            let digit = ((e.0[(shift / 64) as usize] >> (shift % 64)) & 0xf) as usize;
            if digit != 0 {
                acc = self.mul(&acc, &table[digit]);
            }
        }
        acc
    }
}

/// The Jacobi symbol `(a / n)` for a small `a` and an odd `n`.
pub fn jacobi_small<const N: usize>(a: u64, n: &Uint<N>) -> i32 {
    debug_assert!(n.is_odd());
    if n == &Uint::<N>::ONE {
        return 1;
    }
    let mut a = a;
    let mut result = 1i32;
    if a == 0 {
        return 0;
    }
    let n8 = n.0[0] & 7;
    while a & 1 == 0 {
        a >>= 1;
        if n8 == 3 || n8 == 5 {
            result = -result;
        }
    }
    if a == 1 {
        return result;
    }
    // Reciprocity: (a / n) = (n / a) (-1)^((a - 1)(n - 1) / 4), both odd.
    if a & 3 == 3 && n.0[0] & 3 == 3 {
        result = -result;
    }
    result * jacobi_u64(n.rem_u64(a), a)
}

fn jacobi_u64(mut a: u64, mut n: u64) -> i32 {
    let mut result = 1i32;
    a %= n;
    while a != 0 {
        while a & 1 == 0 {
            a >>= 1;
            if n & 7 == 3 || n & 7 == 5 {
                result = -result;
            }
        }
        core::mem::swap(&mut a, &mut n);
        if a & 3 == 3 && n & 3 == 3 {
            result = -result;
        }
        a %= n;
    }
    if n == 1 {
        result
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{jacobi_small, jacobi_u64, Monty, U4};

    #[test]
    fn arithmetic_small_values() {
        let a = U4::from_u64(1_000_003);
        let b = U4::from_u64(999_983);
        assert_eq!(a.wrapping_mul(&b).low_u128(), 1_000_003u128 * 999_983);
        let (q, r) = a.div_rem(&b);
        assert_eq!(q.low_u128(), 1);
        assert_eq!(r.low_u128(), 1_000_003 - 999_983);
        assert_eq!(U4::from_u64(144).isqrt().low_u128(), 12);
        assert_eq!(U4::from_u64(145).isqrt().low_u128(), 12);
        // 3^5 mod 7 = 5
        assert_eq!(
            U4::from_u64(3)
                .pow_mod(&U4::from_u64(5), &U4::from_u64(7))
                .low_u128(),
            5
        );
        let big = U4::pow2(200).wrapping_sub(&U4::ONE);
        let (q, r) = big.div_rem(&U4::pow2(100));
        assert_eq!(q, U4::pow2(100).wrapping_sub(&U4::ONE));
        assert_eq!(r, U4::pow2(100).wrapping_sub(&U4::ONE));
    }

    /// The reference of the word-level division: shift-and-subtract.
    fn div_rem_bits(a: &U4, d: &U4) -> (U4, U4) {
        let mut q = U4::ZERO;
        let mut r = U4::ZERO;
        for i in (0..a.bits()).rev() {
            r = r.shl(1);
            r.0[0] |= (a.0[(i / 64) as usize] >> (i % 64)) & 1;
            if !r.lt(d) {
                r = r.wrapping_sub(d);
                q.0[(i / 64) as usize] |= 1u64 << (i % 64);
            }
        }
        (q, r)
    }

    /// The reference of the Montgomery product: the 512-bit product reduced
    /// bit by bit.
    fn mul_mod_bits(a: &U4, b: &U4, m: &U4) -> U4 {
        let w = a.mul_wide(b);
        let mut r = U4::ZERO;
        for i in (0..512).rev() {
            r = r.shl(1);
            r.0[0] |= (w[(i / 64) as usize] >> (i % 64)) & 1;
            if !r.lt(m) {
                r = r.wrapping_sub(m);
            }
        }
        r
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 ^ (self.0 >> 29)
        }
        fn u4(&mut self, limbs: usize, top_bits: u32) -> U4 {
            let mut x = U4::ZERO;
            for i in 0..limbs {
                x.0[i] = self.next();
            }
            if limbs > 0 && top_bits < 64 {
                x.0[limbs - 1] >>= 64 - top_bits.max(1);
                x.0[limbs - 1] |= 1 << (top_bits.max(1) - 1);
            }
            x
        }
    }

    #[test]
    fn division_matches_the_bit_serial_reference() {
        let mut g = Lcg(7);
        for _ in 0..3000 {
            let (la, ta) = ((1 + g.next() % 4) as usize, (1 + g.next() % 64) as u32);
            let a = g.u4(la, ta);
            let (ld, td) = ((1 + g.next() % 4) as usize, (1 + g.next() % 64) as u32);
            let d = g.u4(ld, td);
            if d.is_zero() {
                continue;
            }
            assert_eq!(a.div_rem(&d), div_rem_bits(&a, &d), "{a:?} / {d:?}");
            assert_eq!(
                a.rem_u64(d.0[0] | 1),
                div_rem_bits(&a, &U4::from_u64(d.0[0] | 1)).1 .0[0]
            );
        }
        let n = U4::pow2(174).wrapping_sub(&U4::from_u64(12345));
        for e in [0u32, 1, 63, 64, 65, 127, 128, 200, 255] {
            let a = U4::pow2(e).wrapping_sub(&U4::ONE);
            assert_eq!(a.div_rem(&n), div_rem_bits(&a, &n));
            assert_eq!(
                n.div_rem(&a.wrapping_add(&U4::ONE)),
                div_rem_bits(&n, &a.wrapping_add(&U4::ONE))
            );
        }
    }

    #[test]
    fn montgomery_matches_the_bit_serial_reference() {
        let mut g = Lcg(11);
        for _ in 0..300 {
            let (lm, tm) = ((1 + g.next() % 4) as usize, (1 + g.next() % 63) as u32);
            let mut m = g.u4(lm, tm);
            m.0[0] |= 1;
            if m == U4::ONE {
                continue;
            }
            let a = g.u4(4, 64).div_rem(&m).1;
            let b = g.u4(4, 64).div_rem(&m).1;
            assert_eq!(
                a.mul_mod(&b, &m),
                mul_mod_bits(&a, &b, &m),
                "{a:?} * {b:?} mod {m:?}"
            );
            let e = U4::from_u64(g.next() % 5000);
            let mut want = U4::ONE;
            for _ in 0..e.0[0] {
                want = mul_mod_bits(&want, &a, &m);
            }
            assert_eq!(a.pow_mod(&e, &m), want, "{a:?}^{e:?} mod {m:?}");
        }
        // A 173-bit modulus of Cornacchia's shape and a 172-bit exponent.
        let n = U4::pow2(174).wrapping_sub(&U4::from_u64(0x3f5a_9b17));
        let e = n.wrapping_sub(&U4::ONE).shr(2);
        let a = U4::from_u64(2);
        let mt = Monty::new(&n);
        let fast = mt.from(&mt.pow(&mt.to(&a), &e));
        let mut slow = U4::ONE;
        let mut base = a;
        for i in 0..e.bits() {
            if (e.0[(i / 64) as usize] >> (i % 64)) & 1 == 1 {
                slow = mul_mod_bits(&slow, &base, &n);
            }
            base = mul_mod_bits(&base, &base, &n);
        }
        assert_eq!(fast, slow);
    }

    #[test]
    fn jacobi_matches_eulers_criterion_on_small_primes() {
        for &p in &[
            5u64, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 1009, 1013, 65537,
        ] {
            let n = U4::from_u64(p);
            let e = U4::from_u64((p - 1) / 2);
            for a in 1..40u64 {
                let euler = U4::from_u64(a).pow_mod(&e, &n);
                let want = if a % p == 0 {
                    0
                } else if euler == U4::ONE {
                    1
                } else {
                    -1
                };
                assert_eq!(jacobi_small(a, &n), want, "({a} / {p})");
            }
        }
        // A composite modulus: the symbol is multiplicative in the modulus.
        let n = U4::from_u64(15);
        for a in 1..15u64 {
            let want = jacobi_u64(a % 3, 3) * jacobi_u64(a % 5, 5);
            assert_eq!(jacobi_small(a, &n), want);
        }
    }
}
