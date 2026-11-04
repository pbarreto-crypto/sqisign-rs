//! The compact format's challenge and the recovery of the dropped response
//! scalar.
//!
//! The challenge is the round-3 hash shape applied to the compact key:
//! `SHAKE256("SQI" || pk_bytes || A_com || msg)`, the first `λ/8` bytes as a
//! little-endian scalar (the specification's `hash_to_challenge` with the
//! compact public key in place of the round-3 one). It is not transmitted:
//! the verifier recomputes it from the transmitted commitment curve.

use super::params::Scalar;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

use crate::fp::{Fp2, FpBackend};

use super::params::HdLevel;

/// The largest challenge of the implemented levels, in bytes.
pub const MAX_CHAL_BYTES: usize = 32;

/// Bytes of the challenge at this level, `λ/8`.
#[inline]
pub fn hd_challenge_len<L: HdLevel>() -> usize {
    L::CHALLENGE_BYTES
}

/// `SHAKE256("SQI" || pk_bytes || A_com || msg)`, `out.len()` bytes.
#[inline]
pub fn hd_challenge<L: FpBackend>(pk_bytes: &[u8], a_com: &Fp2<L>, message: &[u8], out: &mut [u8]) {
    let mut hasher = Shake256::default();
    hasher.update(b"SQI");
    hasher.update(pk_bytes);
    hasher.update(a_com.encode().as_ref());
    hasher.update(message);
    hasher.finalize_xof().read(out);
}

/// Why a compact verification rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdReject {
    /// A curve coefficient that is not a valid Montgomery curve.
    BadCurve,
    /// The challenge does not recompute (unused on the wire path, kept for
    /// callers that carry a challenge).
    ChallengeMismatch,
    /// The challenge isogeny or its basis could not be recovered.
    ChallengeRecovery,
    /// The response images could not be recovered (the pairing discrete
    /// logarithm failed: the commitment and challenge bases are inconsistent).
    ResponseRecovery,
    /// `2^e − q` is not a sum of two squares the verifier can find.
    NormEquation,
    /// The degree is not good: `2^e − q` is not a probable prime congruent to
    /// 1 modulo 4 (the goodness test of the paper's Section 4, repeated by
    /// the verifier).
    DegreeNotGood,
    /// The gluing parameter `m = v_2(a_2)` is out of range.
    GluingBound,
    /// The canonical four-torsion of the commitment side could not be formed.
    CanonicalCom,
    /// The canonical four-torsion of the challenge side could not be formed
    /// (the response images are not a basis of the `2^r`-torsion).
    CanonicalChal,
    /// A half-chain failed (gluing or a product along the chain).
    ChainFailed,
    /// The two half-chains do not meet.
    MiddleCodomainMismatch,
    /// The embedded isogeny is not the response.
    HdImageMismatch,
    /// Wrong length or a non-canonical field element or scalar.
    MalformedInput,
}

#[inline]
fn mask_r(r: u32) -> Scalar {
    Scalar::mask(r)
}

#[inline]
fn mul_r(x: Scalar, y: Scalar, r: u32) -> Scalar {
    (x * y) & mask_r(r)
}

#[inline]
fn inv_2r(a: Scalar, r: u32) -> Scalar {
    debug_assert!(a.is_odd(), "inverse mod 2^r requires odd input");
    a.inv_mod_pow2(r)
}


/// The dropped scalar from `a d − b c ≡ k q (mod 2^r)`: `(c, d)` with the
/// transmitted one of them being `c_or_d` (`c` when `a` is odd, else `d`).
///
/// This is the rule of the paper's Section 4 ("Which entry is dropped"): if
/// `a` is odd, `d ≡ (kq + bc) a^{-1}`; if `a` is even, `c ≡ (ad − kq) b^{-1}`.
/// The rule inverts `a` or `b`, so it is defined only when the entry it
/// inverts is odd. `None` when `a` and `b` are both even, and the signature
/// is rejected: for odd `q`, as every good degree is, no `(c, d)` then
/// satisfies the relation, since `ad − bc` would be even and `kq` is odd.
#[inline]
pub fn recover_response_cd(
    a: Scalar,
    b: Scalar,
    c_or_d: Scalar,
    q: Scalar,
    k: Scalar,
    r: u32,
) -> Option<(Scalar, Scalar)> {
    let m = mask_r(r);
    let (a_r, b_r, cod) = (a & m, b & m, c_or_d & m);
    let kq = mul_r(k & m, q & m, r);
    if a_r.is_odd() {
        let c = cod;
        let bc = mul_r(b_r, c, r);
        let d = mul_r(inv_2r(a_r, r), (kq + bc) & m, r);
        Some((c, d))
    } else if b_r.is_odd() {
        let d = cod;
        let ad = mul_r(a_r, d, r);
        let c = mul_r(inv_2r(b_r, r), (ad - kq) & m, r);
        Some((c, d))
    } else {
        None
    }
}
