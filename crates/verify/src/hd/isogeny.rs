//! A single dimension-4 `(2,2,2,2)`-isogeny step between theta structures.
//!
//! This ports the *plain* `IsogenyDim4` of the sage `Theta_dim4` reference
//! (`isogenies/isogeny_dim4.py`): given a kernel described by four 8-torsion
//! theta points on the domain, it computes the codomain theta structure and
//! evaluates the isogeny on domain points. It does **not** cover the
//! gluing/splitting isogenies (those have a different kernel description and the
//! `special_image` evaluation; they belong to a later phase).
//!
//! # Algorithm
//!
//! Let `K₈ = (K₀, K₁, K₂, K₃)` be the four 8-torsion kernel points and
//! `HSKₖ = H(S(Kₖ))` their squared-dual coordinates (`S` = coordinate-wise
//! square, `H` = Hadamard). The codomain **dual** theta null point `O` is the
//! unique-up-to-scale vector with
//!
//! ```text
//!   O[j ⊕ 2ᵏ] / O[j] = HSKₖ[j ⊕ 2ᵏ] / HSKₖ[j]
//! ```
//!
//! for every index `j` and bit `k` with `HSKₖ[j] ≠ 0`. We pin `O[j₀] = 1` at a
//! root `j₀` and propagate these ratios over a spanning tree of the 4-cube
//! (edges = single-bit flips, the flip of bit `k` measured by kernel point
//! `k`). The codomain (standard) theta null point is then `H(O)`, and the
//! isogeny image is `image(P) = H( H(S(P)) ⊙ O⁻¹ )`.
//!
//! Because the kernel comes from a genuine isogeny, the ratios are consistent
//! (path-independent around every 2-face), so the spanning tree chosen is
//! irrelevant: `O` - hence the codomain - is deterministic, matching the
//! oracle. Comparisons are nevertheless projective (see
//! [`ThetaPointDim4::projective_eq`]).

use crate::{Fp2, FpBackend};

use crate::hd::arith::{hadamard, hadamard_assign, pointwise_mul_assign, to_squared_theta_assign};
use crate::hd::point::{ThetaPointDim4, THETA_DIM4_N as N};
use crate::hd::structure::ThetaStructureDim4;

/// A computed dimension-4 `(2,2,2,2)`-isogeny: its codomain structure plus the
/// precomputed inverse dual theta-null constants used for image evaluation.
#[derive(Clone, Debug)]
pub struct IsogenyDim4<L: FpBackend> {
    codomain: ThetaStructureDim4<L>,
    /// `1 / O[i]` for each non-zero dual null coordinate; `0` where `O[i] = 0`.
    inv_dual_null: [Fp2<L>; N],
    /// `true` for every index where the codomain dual null coordinate is zero.
    dual_null_zero: [bool; N],
    /// `true` if any dual null coordinate is zero (then [`Self::image`] needs
    /// the un-ported `special_image`; see crate notes).
    has_zero_dual: bool,
}

impl<L: FpBackend> IsogenyDim4<L> {
    /// Compute the isogeny from its kernel: four 8-torsion theta points on the
    /// domain (`4·Kₖ` is a basis of the `(2,2,2,2)` kernel).
    ///
    /// Returns `None` if the codomain dual null point cannot be reconstructed
    /// (no root spans the 4-cube). The sage reference falls back to a random
    /// symplectic base change here; that fallback is **not** ported (it is
    /// documented as exceptional in large characteristic). At Level 1 this has
    /// not been observed.
    pub fn from_kernel(k8: &[ThetaPointDim4<L>; 4]) -> Option<Self> {
        // Squared-dual coordinates of each kernel point, one copy each.
        let hsk: [[Fp2<L>; N]; 4] = core::array::from_fn(|k| {
            let mut a = k8[k].coords().clone();
            to_squared_theta_assign(&mut a);
            a
        });

        // Plain step: the dual null and the image weights as products, no inversion,
        // when no coordinate is zero; the inversion path with its zero handling otherwise.
        let (dual_null, inv_dual_null, dual_null_zero, has_zero_dual) =
            match codomain_dual_null_projective(&hsk) {
                Some((o, w)) => (o, w, [false; N], false),
                None => codomain_dual_null_and_inverse(&hsk, &[1, 2, 4, 8])?,
            };

        // Codomain (standard) theta null point is the Hadamard of the dual.
        let codomain_null = ThetaPointDim4::new(hadamard(&dual_null));

        Some(Self {
            codomain: ThetaStructureDim4::new(codomain_null),
            inv_dual_null,
            dual_null_zero,
            has_zero_dual,
        })
    }

    /// The codomain theta structure.
    #[inline]
    pub fn codomain(&self) -> &ThetaStructureDim4<L> {
        &self.codomain
    }

    /// The codomain theta null point.
    #[inline]
    pub fn codomain_null(&self) -> &ThetaPointDim4<L> {
        self.codomain.null_point()
    }

    /// `true` if the codomain has a zero dual theta-null coordinate, in which
    /// case [`Self::image`] returns `None` (the generic image formula does not
    /// apply; `special_image` would be required).
    #[inline]
    pub fn has_zero_dual_null(&self) -> bool {
        self.has_zero_dual
    }

    /// Per-index flags: `true` where the codomain dual null coordinate is zero.
    #[inline]
    pub fn dual_null_zero_mask(&self) -> &[bool; N] {
        &self.dual_null_zero
    }

    /// The image precomputation `1/O` (for reuse without rebuilding the isogeny).
    #[inline]
    pub fn inv_dual_null(&self) -> &[Fp2<L>; N] {
        &self.inv_dual_null
    }

    /// Evaluate the isogeny on a domain theta point:
    /// `image(P) = H( H(S(P)) ⊙ O⁻¹ )`.
    ///
    /// Returns `None` when the codomain has a zero dual null coordinate (see
    /// [`Self::has_zero_dual_null`]); the result is otherwise a theta point on
    /// the codomain.
    #[inline]
    pub fn image(&self, p: &ThetaPointDim4<L>) -> Option<ThetaPointDim4<L>> {
        if self.has_zero_dual {
            return None;
        }
        Some(apply_plain_image(&self.inv_dual_null, p))
    }

    /// [`Self::image`] in place: `p <- image(p)`; `false`, with `p` untouched,
    /// when the codomain has a zero dual null coordinate.
    #[inline]
    pub fn image_assign(&self, p: &mut ThetaPointDim4<L>) -> bool {
        if self.has_zero_dual {
            return false;
        }
        apply_plain_image_assign(&self.inv_dual_null, p);
        true
    }

    /// [`Self::from_kernel`] for a kernel `[K0, K1, PI(K0), PI(K1)]` of the
    /// two-point verifier, the third and fourth points being exact coordinate
    /// permutations of the first two. Squaring and the Hadamard transform
    /// commute with `PI`, so two of the four squared-theta vectors are
    /// permutations of the other two, which gives bit-identical results. The
    /// codomain's dual null point, fixed by `PI` (the paper's Proposition 5),
    /// is computed on the ten orbit representatives, with a check that falls
    /// back to the full computation.
    pub fn from_kernel_pi_paired(k8: &[ThetaPointDim4<L>; 4]) -> Option<Self> {
        use crate::hd::two_point::PI;
        let mut h0 = k8[0].coords().clone();
        to_squared_theta_assign(&mut h0);
        let mut h1 = k8[1].coords().clone();
        to_squared_theta_assign(&mut h1);
        let h2: [Fp2<L>; N] = core::array::from_fn(|a| h0[PI[a]].clone());
        let h3: [Fp2<L>; N] = core::array::from_fn(|a| h1[PI[a]].clone());
        if let Some((dual_null, inv_dual_null)) = codomain_dual_null_direct(&h0, &h1) {
            let codomain_null = ThetaPointDim4::new(
                crate::hd::two_point::hadamard_symmetric(&dual_null),
            );
            return Some(Self {
                codomain: ThetaStructureDim4::new(codomain_null),
                inv_dual_null,
                dual_null_zero: [false; N],
                has_zero_dual: false,
            });
        }
        let hsk = [h0, h1, h2, h3];
        let (dual_null, inv_dual_null, dual_null_zero, has_zero_dual) =
            match codomain_dual_null_symmetric(&hsk) {
                Some((o, w)) => (o, w, [false; N], false),
                None => match codomain_dual_null_projective(&hsk) {
                    Some((o, w)) => (o, w, [false; N], false),
                    None => codomain_dual_null_and_inverse(&hsk, &[1, 2, 4, 8])?,
                },
            };
        let codomain_null = ThetaPointDim4::new(hadamard(&dual_null));
        Some(Self {
            codomain: ThetaStructureDim4::new(codomain_null),
            inv_dual_null,
            dual_null_zero,
            has_zero_dual,
        })
    }

    /// The codomain structure, moved out (the chain continues on it).
    #[inline]
    pub fn into_codomain(self) -> ThetaStructureDim4<L> {
        self.codomain
    }
}

/// [`apply_plain_image`] in place on the point's coordinates.
#[inline]
pub fn apply_plain_image_assign<L: FpBackend>(
    inv_dual_null: &[Fp2<L>; N],
    p: &mut ThetaPointDim4<L>,
) {
    let a = p.coords_mut();
    to_squared_theta_assign(a);
    pointwise_mul_assign(a, inv_dual_null);
    hadamard_assign(a);
}

/// Apply a *plain*-step image map given its precomputed `1/O`:
/// `image(P) = H( H(S(P)) ⊙ O⁻¹ )`. Lets a caller that already holds the image
/// precomputation (e.g. the strategy chain) push a point through a plain step
/// without rebuilding the isogeny. Only valid for zero-free `O` (plain steps).
#[inline]
pub fn apply_plain_image<L: FpBackend>(
    inv_dual_null: &[Fp2<L>; N],
    p: &ThetaPointDim4<L>,
) -> ThetaPointDim4<L> {
    let mut a = p.coords().clone();
    to_squared_theta_assign(&mut a);
    pointwise_mul_assign(&mut a, inv_dual_null);
    hadamard_assign(&mut a);
    ThetaPointDim4::new(a)
}

/// The plain step's dual null `O` and weights `1/O` from the spanning tree of the 4-cube, up
/// to common scalars (products, no inversion); `None` when no root spans it or a `p_num` is zero.
#[allow(clippy::type_complexity)]
fn dual_null_tree<L: FpBackend>(
    hsk: &[[Fp2<L>; N]; 4],
) -> Option<([Fp2<L>; N], [Fp2<L>; N])> {
    let dirs = [1usize, 2, 4, 8];
    for j0 in 0..N {
        if (0..4).all(|k| bool::from(hsk[k][j0].ct_is_zero())) {
            continue;
        }
        let mut covered = [false; N];
        let mut queue = [0usize; N];
        let (mut head, mut tail) = (0usize, 0usize);
        let mut edge_j = [0usize; N];
        let mut edge_jpk = [0usize; N];
        let mut edge_k = [0usize; N];
        let mut ne = 0usize;
        covered[j0] = true;
        queue[tail] = j0;
        tail += 1;
        while head < tail {
            let j = queue[head];
            head += 1;
            for (k, hsk_k) in hsk.iter().enumerate() {
                let jpk = j ^ dirs[k];
                if !covered[jpk] && !bool::from(hsk_k[j].ct_is_zero()) {
                    edge_j[ne] = j;
                    edge_jpk[ne] = jpk;
                    edge_k[ne] = k;
                    ne += 1;
                    covered[jpk] = true;
                    queue[tail] = jpk;
                    tail += 1;
                }
            }
        }
        if !covered.iter().all(|&c| c) {
            continue;
        }
        let mut p_num: [Fp2<L>; N] = core::array::from_fn(|_| Fp2::zero());
        let mut p_den: [Fp2<L>; N] = core::array::from_fn(|_| Fp2::zero());
        p_num[j0] = Fp2::one();
        p_den[j0] = Fp2::one();
        for e in 0..ne {
            let (j, jpk, k) = (edge_j[e], edge_jpk[e], edge_k[e]);
            p_num[jpk] = p_num[j].mul(&hsk[k][jpk]);
            p_den[jpk] = p_den[j].mul(&hsk[k][j]);
        }
        if p_num.iter().any(|x| bool::from(x.ct_is_zero())) {
            return None;
        }
        return Some((p_num, p_den));
    }
    None
}

/// The dual null point `O` of the codomain and a vector proportional to its
/// inverse, from the squared-theta kernel coordinates `hsk`, without
/// inversions: `None` if a coordinate is zero. `O = p_num (.) others(p_den)`,
/// and `1/O` up to a scalar is `others(O)` itself.
fn codomain_dual_null_projective<L: FpBackend>(
    hsk: &[[Fp2<L>; N]; 4],
) -> Option<([Fp2<L>; N], [Fp2<L>; N])> {
    let (p_num, p_den) = dual_null_tree(hsk)?;
    let others_den = crate::hd::field::products_of_others(&p_den);
    let o: [Fp2<L>; N] = core::array::from_fn(|j| p_num[j].mul(&others_den[j]));
    let w = crate::hd::field::products_of_others(&o);
    Some((o, w))
}

/// The inverse dual null point of the codomain, directly from the squared-theta
/// vectors `h0`, `h1` of the two pushed kernel points, as products along a
/// fixed path through the ten orbit representatives of `PI`: the formula of
/// MIKE's reference implementation (`solve_HIIP_weil`), carried over to our
/// indexing by the relabeling of index bits 0 -> 2, 1 -> 3, 2 -> 1, 3 -> 0,
/// which conjugates MIKE's permutation to `PI` and its kernel directions to
/// ours. About 26 multiplications. It uses the symmetry of the dual null point
/// (the paper's Proposition 5) without checking it, as the two-point chain
/// relies on Theorem 2. Returns `(O, 1/O)` up to scalars, the dual null point
/// by a ten-term pseudo-inversion; `None` if a coordinate vanishes, in which
/// case the caller falls back to the checked computation.
fn codomain_dual_null_direct<L: FpBackend>(
    h0: &[Fp2<L>; N],
    h1: &[Fp2<L>; N],
) -> Option<([Fp2<L>; N], [Fp2<L>; N])> {
    let r1 = h0[3].clone();
    let r2 = r1.mul(&h0[9]);
    let r3 = r2.mul(&h1[8]);
    let r4 = r3.mul(&h0[10]);
    let r5 = r4.mul(&h0[14]);
    let r6 = r5.mul(&h1[15]);
    let r7 = r6.mul(&h1[7]);
    let r8 = r7.mul(&h0[5]);
    let b8 = h0[0].clone();
    let b7 = b8.mul(&h0[4]);
    let b6 = b7.mul(&h1[5]);
    let b5 = b6.mul(&h1[13]);
    let b4 = b5.mul(&h0[15]);
    let b3 = b4.mul(&h0[11]);
    let b2 = b3.mul(&h1[10]);
    let b1 = b2.mul(&h0[8]);
    let m0 = r8.mul(&h0[1]);
    let m12 = h0[2].mul(&h0[9]).mul(&b2);
    let m5 = r1.mul(&b1);
    let m1 = r2.mul(&b2);
    let m9 = r3.mul(&b3);
    let m11 = r4.mul(&b4);
    let m15 = r5.mul(&b5);
    let m14 = r6.mul(&b6);
    let m6 = r7.mul(&b7);
    let m4 = r8.mul(&b8);
    let w: [Fp2<L>; N] = [
        m0.clone(), m4.clone(), m1.clone(), m12.clone(), m4.clone(), m6.clone(), m5.clone(), m14.clone(), m1.clone(), m5.clone(), m9.clone(), m11.clone(), m12.clone(), m14.clone(), m11.clone(), m15.clone(),
    ];
    if w.iter().any(|x| bool::from(x.ct_is_zero())) {
        return None;
    }
    let o = crate::hd::two_point::inverses_of_symmetric(&w);
    if o.iter().any(|x| bool::from(x.ct_is_zero())) {
        return None;
    }
    Some((o, w))
}

/// [`codomain_dual_null_projective`] for a codomain whose dual null point is
/// fixed by `PI` (the paper's Proposition 5): `O` is computed on the ten orbit
/// representatives and copied to the partners. The symmetry is checked, not
/// assumed: for each pair `(a, PI[a])`, `O[a]` and `O[PI[a]]` are compared
/// through one cross-multiplication of the tree's numerators and
/// denominators, and any mismatch returns `None`, so that the caller falls
/// back to the full computation.
fn codomain_dual_null_symmetric<L: FpBackend>(
    hsk: &[[Fp2<L>; N]; 4],
) -> Option<([Fp2<L>; N], [Fp2<L>; N])> {
    use crate::hd::two_point::{PI_PAIRS, PI_REPS, PI_REP_INDEX};
    let (p_num, p_den) = dual_null_tree(hsk)?;
    for &(a, b) in PI_PAIRS.iter() {
        if !bool::from(p_num[a].mul(&p_den[b]).ct_equal(&p_num[b].mul(&p_den[a]))) {
            return None;
        }
    }
    let den: [Fp2<L>; 10] = core::array::from_fn(|i| p_den[PI_REPS[i]].clone());
    let others_den = crate::hd::field::products_of_others(&den);
    let o_reps: [Fp2<L>; 10] = core::array::from_fn(|i| p_num[PI_REPS[i]].mul(&others_den[i]));
    let w_reps = crate::hd::field::products_of_others(&o_reps);
    let o = core::array::from_fn(|a| o_reps[PI_REP_INDEX[a]].clone());
    let w = core::array::from_fn(|a| w_reps[PI_REP_INDEX[a]].clone());
    Some((o, w))
}

/// Reconstruct a codomain dual theta null point `O` from kernel points'
/// squared-dual coordinates and their index-flip directions.
///
/// `hsk[k] = H(S(K_k))` for each kernel point, and `dirs[k]` is the index XOR
/// that translation by `K_k` induces (`2ᵏ` for a single-bit kernel generator,
/// or e.g. `3 = 2⁰⊕2¹` for a diagonal generator used by the gluing). Pin
/// `O[j₀] = 1` at a root and propagate `O[j⊕dirs[k]] = O[j]·HSKₖ[j⊕dirs[k]]/
/// HSKₖ[j]` over a spanning tree (an edge is usable iff its denominator
/// `HSKₖ[j] ≠ 0`; the numerator may be zero, yielding a zero `O` coordinate).
/// Tries successive roots until one spans all 16 indices; `None` if none does.
///
/// Shared by the plain step (`dirs = [1,2,4,8]`) and the gluing step
/// (`dirs = [1,2,4,8,3]`, whose diagonal edge spans the cube despite the zero
/// dual coordinates).
#[allow(clippy::type_complexity)] // (O, 1/O, zero-mask, has-zero) - a small tuple.
pub(crate) fn codomain_dual_null_and_inverse<L: FpBackend>(
    hsk: &[[Fp2<L>; N]],
    dirs: &[usize],
) -> Option<([Fp2<L>; N], [Fp2<L>; N], [bool; N], bool)> {
    debug_assert_eq!(hsk.len(), dirs.len());
    for j0 in 0..N {
        // The root needs at least one usable outgoing edge.
        if (0..hsk.len()).all(|k| bool::from(hsk[k][j0].ct_is_zero())) {
            continue;
        }

        // Pass 1 - BFS over the 4-cube using only the (value-independent) zero
        // pattern, recording the spanning-tree edges in discovery order with
        // their numerators/denominators. No inversions yet.
        let mut covered = [false; N];
        let mut queue = [0usize; N];
        let (mut head, mut tail) = (0usize, 0usize);
        let mut edge_j = [0usize; N];
        let mut edge_jpk = [0usize; N];
        let mut num: [Fp2<L>; N] = core::array::from_fn(|_| Fp2::zero());
        let mut den: [Fp2<L>; N] = core::array::from_fn(|_| Fp2::zero());
        let mut ne = 0usize;

        covered[j0] = true;
        queue[tail] = j0;
        tail += 1;
        while head < tail {
            let j = queue[head];
            head += 1;
            for (k, hsk_k) in hsk.iter().enumerate() {
                let jpk = j ^ dirs[k];
                if !covered[jpk] && !bool::from(hsk_k[j].ct_is_zero()) {
                    edge_j[ne] = j;
                    edge_jpk[ne] = jpk;
                    num[ne] = hsk_k[jpk].clone();
                    den[ne] = hsk_k[j].clone();
                    ne += 1;
                    covered[jpk] = true;
                    queue[tail] = jpk;
                    tail += 1;
                }
            }
        }
        if !covered.iter().all(|&c| c) {
            continue;
        }

        // Pass 2 - propagate the path products (no inversions):
        // `O[j] = p_num[j] / p_den[j]` where `p_num`/`p_den` are the running
        // products of the edge numerators/denominators from the root.
        let mut p_num: [Fp2<L>; N] = core::array::from_fn(|_| Fp2::zero());
        let mut p_den: [Fp2<L>; N] = core::array::from_fn(|_| Fp2::zero());
        p_num[j0] = Fp2::one();
        p_den[j0] = Fp2::one();
        for e in 0..ne {
            let (j, jpk) = (edge_j[e], edge_jpk[e]);
            p_num[jpk] = p_num[j].mul(&num[e]);
            p_den[jpk] = p_den[j].mul(&den[e]);
        }

        // A single batched inversion over all `p_den` (always non-zero) plus the
        // non-zero `p_num`. Then `O = p_num·(1/p_den)` and `1/O = p_den·(1/p_num)`.
        let mut batch: [Fp2<L>; 2 * N] = core::array::from_fn(|_| Fp2::zero());
        batch[..N].clone_from_slice(&p_den[..N]);
        let mut idx = [0usize; N];
        let mut cnt = 0;
        let mut zero_mask = [false; N];
        let mut has_zero = false;
        for j in 0..N {
            // `p_num[j] == 0` ⇔ a zero edge numerator on the path ⇔ `O[j] = 0`.
            if bool::from(p_num[j].ct_is_zero()) {
                zero_mask[j] = true;
                has_zero = true;
            } else {
                batch[N + cnt] = p_num[j].clone();
                idx[cnt] = j;
                cnt += 1;
            }
        }
        let nb = N + cnt;
        let mut t1: [Fp2<L>; 2 * N] = core::array::from_fn(|_| Fp2::zero());
        let mut t2: [Fp2<L>; 2 * N] = core::array::from_fn(|_| Fp2::zero());
        crate::hd::field::batched_inv(&mut batch[..nb], &mut t1[..nb], &mut t2[..nb]);

        let mut o: [Fp2<L>; N] = core::array::from_fn(|_| Fp2::zero());
        let mut inv_o: [Fp2<L>; N] = core::array::from_fn(|_| Fp2::zero());
        for j in 0..N {
            // `O[j] = p_num[j] / p_den[j]` (zero where `p_num[j] = 0`).
            o[j] = p_num[j].mul(&batch[j]);
        }
        for e in 0..cnt {
            let j = idx[e];
            inv_o[j] = p_den[j].mul(&batch[N + e]); // `1/O[j] = p_den[j] / p_num[j]`
        }
        return Some((o, inv_o, zero_mask, has_zero));
    }
    None
}
