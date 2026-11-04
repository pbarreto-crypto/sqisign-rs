//! The parameters of the compact format per prime (`docs/COMPACT_R3.md`):
//! the embedding exponent `e` (the response degree is `q < 2^e`), the
//! torsion `r = ⌈e/2⌉ + 2` the half-chains use and the response scalars
//! are transmitted modulo, the challenge length `λ`, and the library's
//! non-residue tables for the basis-from-hint convention. Levels I, III and V.

use crate::fp::FpBackend;
use super::uint::Uint;

/// The integer type of the scalars modulo `2^r` (the transmitted action and
/// the starting matrices): 192 bits, enough for every level (`r = 89, 134,
/// 175`). Scalar arithmetic is a negligible share of a verification, so one
/// width serves all levels.
pub type Scalar = Uint<3>;

/// The integer type of a level's response degree `q < 2^e`: what the wire
/// code, the norm equation and the goodness test need. Implemented for every
/// [`Uint`]; each level picks the narrowest width that holds `2^e`.
pub trait DegreeInt: Copy + Eq + core::fmt::Debug {
    /// From little-endian bytes.
    fn from_le_bytes(b: &[u8]) -> Self;
    /// From little-endian limbs (missing ones are zero).
    fn from_le_limbs(l: &[u64]) -> Self;
    /// Writes the low `out.len()` bytes, little-endian.
    fn write_le_bytes(&self, out: &mut [u8]);
    /// The width in limbs.
    fn limbs() -> usize;
    /// The number of significant bits.
    fn bits(&self) -> u32;
    /// The low 192 bits, as a [`Scalar`].
    fn to_scalar(&self) -> Scalar;
    /// The verifier's goodness test ([`super::primality::is_good_degree`]).
    fn is_good_degree(&self, e: u32) -> bool;
    /// `2^e − self = a1^2 + a2^2` ([`super::kani::norm_equation_2f_minus_q`]).
    fn norm_equation(&self, e: u32) -> Option<(Self, Self)>;
}

impl<const N: usize> DegreeInt for Uint<N> {
    fn from_le_bytes(b: &[u8]) -> Self {
        Uint::from_le_bytes(b)
    }
    fn from_le_limbs(l: &[u64]) -> Self {
        Uint::from_le_limbs(l)
    }
    fn write_le_bytes(&self, out: &mut [u8]) {
        Uint::write_le_bytes(self, out)
    }
    fn limbs() -> usize {
        N
    }
    fn bits(&self) -> u32 {
        Uint::bits(self)
    }
    fn to_scalar(&self) -> Scalar {
        self.resize()
    }
    fn is_good_degree(&self, e: u32) -> bool {
        super::primality::is_good_degree(e, self)
    }
    fn norm_equation(&self, e: u32) -> Option<(Self, Self)> {
        super::kani::norm_equation_2f_minus_q(e, self)
    }
}
use crate::params::{P324_3, P500_27, P664_17};
use crate::precomp::PrimePrecomp;

use super::nqr_tables_p324_3::{NQR_TABLE_P324_3, Z_NQR_TABLE_P324_3};
use super::nqr_tables_p500_27::{NQR_TABLE_P500_27, Z_NQR_TABLE_P500_27};
use super::nqr_tables_p664_17::{NQR_TABLE_P664_17, Z_NQR_TABLE_P664_17};

/// A prime with compact-format parameters.
pub trait HdLevel: FpBackend + PrimePrecomp {
    /// `λ`: the challenge isogeny has degree `2^LAMBDA`.
    const LAMBDA: u32;
    /// `e`: the response degree is below `2^E_EMBED`.
    const E_EMBED: u32;
    /// `r = ⌈e/2⌉ + 2`: the torsion of the half-chains, the scalars' modulus.
    const R: u32;
    /// `⌈e/2⌉`, the length of the first half-chain.
    const E1: u32;
    /// Bytes of one encoded `F_p^2` element (`2 FP`).
    const FP2_BYTES: usize;
    /// Bytes of the response degree on the wire, `⌈e/8⌉`.
    const Q_BYTES: usize;
    /// Bytes of one response scalar on the wire, `⌈r/8⌉`.
    const SCALAR_BYTES: usize;
    /// Bytes of the challenge, `λ/8`.
    const CHALLENGE_BYTES: usize;
    /// The integer type of the response degree, the narrowest that holds
    /// `2^e`.
    type Degree: DegreeInt;
    /// The twenty quadratic non-residues, canonical encodings.
    fn nqr_table() -> [&'static [u8]; 20];
    /// The twenty squares `x` with `x − 1` a non-residue, canonical encodings.
    fn z_nqr_table() -> [&'static [u8]; 20];
    /// The odd cofactor `(p + 1) / 2^f` as limbs, and its bit length.
    fn cofactor() -> ([u64; 1], usize) {
        ([Self::COFACTOR], Self::COFACTOR_BITLENGTH)
    }
}

/// The wire sizes of a level.
pub const fn sig_wire_bytes<L: HdLevel>() -> usize {
    L::FP2_BYTES + L::Q_BYTES + 3 * L::SCALAR_BYTES + 2
}

/// The public key: `A_pk` and two hint bytes.
pub const fn pk_wire_bytes<L: HdLevel>() -> usize {
    L::FP2_BYTES + 2
}

impl HdLevel for P324_3 {
    const LAMBDA: u32 = 128;
    const E_EMBED: u32 = 174;
    const R: u32 = 89;
    const E1: u32 = 87;
    const FP2_BYTES: usize = 82;
    const Q_BYTES: usize = 22;
    const SCALAR_BYTES: usize = 12;
    const CHALLENGE_BYTES: usize = 16;
    type Degree = Uint<3>;
    fn nqr_table() -> [&'static [u8]; 20] {
        core::array::from_fn(|i| &NQR_TABLE_P324_3[i][..])
    }
    fn z_nqr_table() -> [&'static [u8]; 20] {
        core::array::from_fn(|i| &Z_NQR_TABLE_P324_3[i][..])
    }
}

impl HdLevel for P500_27 {
    const LAMBDA: u32 = 192;
    const E_EMBED: u32 = 264;
    const R: u32 = 134;
    const E1: u32 = 132;
    const FP2_BYTES: usize = 128;
    const Q_BYTES: usize = 33;
    const SCALAR_BYTES: usize = 17;
    const CHALLENGE_BYTES: usize = 24;
    type Degree = Uint<5>;
    fn nqr_table() -> [&'static [u8]; 20] {
        core::array::from_fn(|i| &NQR_TABLE_P500_27[i][..])
    }
    fn z_nqr_table() -> [&'static [u8]; 20] {
        core::array::from_fn(|i| &Z_NQR_TABLE_P500_27[i][..])
    }
}

impl HdLevel for P664_17 {
    const LAMBDA: u32 = 256;
    const E_EMBED: u32 = 346;
    const R: u32 = 175;
    const E1: u32 = 173;
    const FP2_BYTES: usize = 168;
    const Q_BYTES: usize = 44;
    const SCALAR_BYTES: usize = 22;
    const CHALLENGE_BYTES: usize = 32;
    type Degree = Uint<6>;
    fn nqr_table() -> [&'static [u8]; 20] {
        core::array::from_fn(|i| &NQR_TABLE_P664_17[i][..])
    }
    fn z_nqr_table() -> [&'static [u8]; 20] {
        core::array::from_fn(|i| &Z_NQR_TABLE_P664_17[i][..])
    }
}

/// The largest signature of the implemented levels (level V).
pub const MAX_SIG_WIRE_BYTES: usize = sig_wire_bytes::<P664_17>();
/// The largest public key of the implemented levels (level V).
pub const MAX_PK_WIRE_BYTES: usize = pk_wire_bytes::<P664_17>();
