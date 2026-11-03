//! Phase 5b.6 - the dimension-4 optimal-strategy chain loop.
//!
//! A SQIsignHD half-chain `F1` is one gluing step (Phase 4) followed by a run of
//! `L` plain `(2,2,2,2)`-steps. The gluing exposes, on its codomain, four
//! full-order (`2^{L+2}`) kernel generators; the rest of the chain is *derived*
//! from them by an optimal-strategy walk that, at each step, doubles the saved
//! kernel bases down to the order-8 kernel of the current step, computes the
//! plain isogeny ([`IsogenyDim4::from_kernel`]), and pushes the remaining bases
//! through it ([`IsogenyDim4::image`]). This is the dimension-4 analogue of the
//! dim-2 `IsogenyChainDim2` loop and of `isogeny_chain_dim4.py`'s plain part -
//! the bookkeeping that turns a single kernel basis into all `L` per-step
//! kernels (the ~938 doublings / ~1882 pushforwards of a Level-1 half-chain).
//!
//! # Strategy-independence (why a self-consistent completion is fine here)
//!
//! The codomain of step `k` and its order-8 kernel are determined by the chain
//! and the position `k`, **not** by the strategy (which only chooses the
//! doubling/pushforward *path*). So any valid optimal strategy reproduces the
//! same per-step kernels and codomains. Likewise, the per-step kernels are the
//! same subgroup generators regardless of the symplectic completion used to
//! build the starting basis (5b.4 decision) - different completions give the
//! same chain up to a global symplectic change of basis, so the middle-codomain
//! match (the accept/reject invariant) is preserved.
//!
//! # `alloc`
//!
//! The optimal-strategy walk keeps a stack of intermediate kernel bases (depth
//! and size data-dependent, bounded by the chain length). It therefore uses
//! heap `Vec`s - the crate's only `alloc` use, off the constant-time path.

use alloc::vec;
use alloc::vec::Vec;

use crate::{Fp2, FpBackend};

use crate::hd::isogeny::IsogenyDim4;
use crate::hd::point::{ThetaPointDim4, THETA_DIM4_N};
use crate::hd::profile::{NoProfiler, Profiler, SLOT_PLAIN_CODOMAIN, SLOT_PLAIN_PUSHES};
use crate::hd::structure::ThetaStructureDim4;

/// Doubling-to-image cost ratio fed to [`optimised_strategy`]. Per point, a
/// doubling costs two pointwise squarings, two pointwise multiplications and two
/// Hadamard transforms (32 squarings, 32 multiplications and 128 additions in
/// `Fp2`), an image one of each (16, 16 and 128), so the ratio lies between 1 and
/// 2 and depends on the cost of an addition; measured in the push slot of the
/// two-point verifier, a doubling is about 1.39 images (15.7 against 11.3
/// kilocycles). The optimum is flat near this value: for the level-I half-chains,
/// the strategy optimized for 1.4 costs at most 0.07 % more than the optimal one
/// for any true ratio between 1.2 and 1.6.
pub(crate) const STRATEGY_MUL_C: f64 = 1.4;

/// The cost of a subtree of `i` leaves split as `b` doublings to the left and
/// `i - b` evaluations to the right, given the optimal costs `c` of smaller
/// subtrees.
#[inline]
fn split_cost(c: &[f64], i: usize, b: usize, mul_c: f64) -> f64 {
    c[i - b] + c[b] + (b as f64) * mul_c + (i - b) as f64
}

/// Optimal `(2,2)`-chain strategy of `n` leaves, with doubling cost `mul_c`
/// relative to a unit isogeny evaluation. Returns the `n - 1` internal nodes
/// (each the number of leaves to the right of that node), in the depth-first
/// order the chain loop walks.
///
/// The optimal cost obeys `C(1) = 0` and `C(i) = min_{1 <= b < i} C(i - b) +
/// C(b) + b·mul_c + (i - b)` (De Feo, Jao and Plût). The function of `b` being
/// minimized has a single minimum (or two adjacent equal ones), so it is found
/// by binary search on consecutive values: Algorithm 5.2 of Zanon, Simplicio,
/// Pereira, Doliskani and Barreto, "Faster isogeny-based compressed key
/// agreement", which takes `O(n log n)` operations instead of `O(n^2)`. Only the
/// optimal split of each subtree size is stored, and the strategy is read off
/// in one pass.
///
/// Any valid strategy yields the same isogeny chain; the cost model only
/// affects efficiency. The gluing step is not part of the strategy: the chain
/// loop starts after it.
pub fn optimised_strategy(n: usize, mul_c: f64) -> Vec<u32> {
    if n <= 1 {
        return Vec::new();
    }
    let mut c = vec![0.0_f64; n + 1];
    let mut split = vec![0_u32; n + 1];
    for i in 2..=n {
        let (mut j, mut z) = (1_usize, i - 1);
        while j < z {
            let m = j + (z - j) / 2;
            if split_cost(&c, i, m, mul_c) <= split_cost(&c, i, m + 1, mul_c) {
                z = m;
            } else {
                j = m + 1;
            }
        }
        c[i] = split_cost(&c, i, j, mul_c);
        split[i] = j as u32;
    }
    // Pre-order walk: a node of `i` leaves emits its split `b`, then the
    // subtree of `i - b` leaves, then the subtree of `b` leaves.
    let mut out = Vec::with_capacity(n - 1);
    let mut stack = vec![n];
    while let Some(i) = stack.pop() {
        if i <= 1 {
            continue;
        }
        let b = split[i] as usize;
        out.push(b as u32);
        stack.push(b);
        stack.push(i - b);
    }
    out
}

#[cfg(test)]
mod strategy_tests {
    use super::*;

    /// The quadratic dynamic program, as a reference.
    fn reference_costs(n: usize, mul_c: f64) -> Vec<f64> {
        let mut c = vec![0.0_f64; n + 1];
        for i in 2..=n {
            c[i] = (1..i)
                .map(|b| split_cost(&c, i, b, mul_c))
                .fold(f64::INFINITY, f64::min);
        }
        c
    }

    /// The cost of a strategy list under the same model.
    fn strategy_cost(s: &[u32], n: usize, mul_c: f64) -> f64 {
        fn go(s: &[u32], pos: &mut usize, i: usize, mul_c: f64) -> f64 {
            if i <= 1 {
                return 0.0;
            }
            let b = s[*pos] as usize;
            *pos += 1;
            let here = (b as f64) * mul_c + (i - b) as f64;
            let left = go(s, pos, i - b, mul_c);
            here + left + go(s, pos, b, mul_c)
        }
        let mut pos = 0;
        let cost = go(s, &mut pos, n, mul_c);
        assert_eq!(pos, s.len(), "strategy list fully consumed");
        cost
    }

    #[test]
    fn binary_search_matches_the_quadratic_program() {
        for &mul_c in &[1.0, 1.2, 1.39, 1.4, 1.6, 2.0, 3.0] {
            let reference = reference_costs(200, mul_c);
            for n in 1..=200 {
                let s = optimised_strategy(n, mul_c);
                assert_eq!(s.len(), n.saturating_sub(1));
                let cost = strategy_cost(&s, n, mul_c);
                assert!((cost - reference[n]).abs() < 1e-9, "n = {n}, mul_c = {mul_c}");
            }
        }
    }
}

/// The result of [`run_strategy_chain`]: the per-step order-8 kernels derived by
/// the strategy walk (one `[_; 4]` per plain step, in chain order) and the
/// corresponding per-step codomain theta null points.
pub struct StrategyChain<L: FpBackend> {
    pub kernels: Vec<[ThetaPointDim4<L>; 4]>,
    pub codomains: Vec<ThetaPointDim4<L>>,
    /// Per-step image precomputation `1/O` (in chain order). Lets a caller push
    /// a point through the forward chain (`apply_plain_image`) without rebuilding
    /// the isogenies - used by the stage-6 HD-image evaluation.
    pub image_precomp: Vec<[Fp2<L>; THETA_DIM4_N]>,
}

impl<L: FpBackend> StrategyChain<L> {
    /// The last codomain (the half-chain's middle theta null point), if any.
    #[inline]
    pub fn last_codomain(&self) -> Option<&ThetaPointDim4<L>> {
        self.codomains.last()
    }
}

/// Derive an `L`-step plain `(2,2,2,2)`-chain from a single full-order kernel
/// basis by walking the optimal strategy.
///
/// * `start` is the theta structure the `basis` lives on (the gluing codomain).
/// * `basis` are four kernel generators of order `2^{L+2}` on `start`.
/// * `l` is the number of plain steps.
///
/// Returns the per-step kernels and codomains, or `None` if any step's codomain
/// is not computable (the non-suitable-null case Phase 4 would handle; not
/// expected at Level 1).
pub fn run_strategy_chain<L: FpBackend>(
    start: &ThetaStructureDim4<L>,
    basis: &[ThetaPointDim4<L>; 4],
    l: usize,
) -> Option<StrategyChain<L>> {
    run_strategy_chain_prof(start, basis, l, &mut NoProfiler)
}

/// [`run_strategy_chain`] with a [`Profiler`]: the doublings and the images
/// of the saved bases go to `SLOT_PLAIN_PUSHES`, `IsogenyDim4::from_kernel`
/// and the next structure's `precompute` to `SLOT_PLAIN_CODOMAIN`. With
/// [`NoProfiler`] the timing calls are empty.
pub fn run_strategy_chain_prof<L: FpBackend, P: Profiler>(
    start: &ThetaStructureDim4<L>,
    basis: &[ThetaPointDim4<L>; 4],
    l: usize,
    prof: &mut P,
) -> Option<StrategyChain<L>> {
    if l == 0 {
        return Some(StrategyChain {
            kernels: Vec::new(),
            codomains: Vec::new(),
            image_precomp: Vec::new(),
        });
    }
    let strategy = optimised_strategy(l, STRATEGY_MUL_C);

    let mut strat_idx = 0usize;
    let mut level: Vec<u32> = vec![0];
    let mut kernel_elements: Vec<[ThetaPointDim4<L>; 4]> = vec![basis.clone()];
    // The doubling formula needs the structure's precomputed inverses; a
    // structure with a non-suitable null cannot be doubled on (the base-change
    // fallback, out of scope - return `None`).
    let mut cur = start.clone();
    let mut cur_ready = cur.precompute();

    let mut kernels: Vec<[ThetaPointDim4<L>; 4]> = Vec::with_capacity(l);
    let mut codomains: Vec<ThetaPointDim4<L>> = Vec::with_capacity(l);
    let mut image_precomp: Vec<[Fp2<L>; THETA_DIM4_N]> = Vec::with_capacity(l);

    for k in 0..l {
        let mut prev: u32 = level.iter().sum();
        let target = (l - 1 - k) as u32;
        let t0 = prof.now();
        while prev != target {
            if !cur_ready {
                return None; // non-suitable null met a required doubling
            }
            let s = strategy[strat_idx];
            level.push(s);
            prev += s;
            // The doublings run in place on a copy of the top basis.
            let mut next = kernel_elements
                .last()
                .expect("non-empty kernel stack")
                .clone();
            for pt in next.iter_mut() {
                cur.double_iter_assign(pt, s);
            }
            kernel_elements.push(next);
            strat_idx += 1;
        }
        let t1 = prof.now();
        prof.add(SLOT_PLAIN_PUSHES, t1 - t0);

        // The descended top-of-stack is the order-8 kernel of this step.
        let ker = kernel_elements
            .last()
            .expect("non-empty kernel stack")
            .clone();
        let iso = IsogenyDim4::from_kernel(&ker)?;
        let t2 = prof.now();
        prof.add(SLOT_PLAIN_CODOMAIN, t2 - t1);
        kernels.push(ker);
        codomains.push(iso.codomain_null().clone());
        image_precomp.push(iso.inv_dual_null().clone());

        kernel_elements.pop();
        level.pop();

        // Push the remaining saved bases through the isogeny, in place.
        for elt in kernel_elements.iter_mut() {
            for slot in elt.iter_mut() {
                if !iso.image_assign(slot) {
                    return None;
                }
            }
        }
        let t3 = prof.now();
        prof.add(SLOT_PLAIN_PUSHES, t3 - t2);
        cur = iso.into_codomain();
        cur_ready = cur.precompute();
        prof.add(SLOT_PLAIN_CODOMAIN, prof.now() - t3);
    }

    Some(StrategyChain {
        kernels,
        codomains,
        image_precomp,
    })
}
