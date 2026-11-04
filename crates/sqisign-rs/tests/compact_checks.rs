//! The two verifier checks of the paper's Section 4 that the compact verifier
//! must make before the chain: the dropped matrix entry is recovered only when
//! the entry the rule inverts is odd (a signature with a and b both even is
//! rejected), and the goodness test on q is repeated (2^e - q must be a
//! probable prime congruent to 1 modulo 4). Feature `compact`. Run with:
//!   cargo test --release -p sqisign-rs --features compact --test compact_checks
#![cfg(feature = "compact")]

use sqisign_rs::compact::{compact_keygen, compact_public_key_to_bytes, compact_sign};
use sqisign_rs::mp::ShakeRng;
use sqisign_rs::sqisign::level1;
use sqisign_verify::hd::primality::is_good_degree;
use sqisign_verify::hd::Degree;
use sqisign_verify::hd::{hd_verify_bytes, HdReject};
use sqisign_verify::P324_3;

const N: usize = sqisign_rs::precomp::p324_3::IBZ_NLIMBS;

// Level-I wire layout: commitment curve (82 bytes), q (22, little-endian),
// a, b, c-or-d (12 each, little-endian), two hint bytes.
const Q_AT: usize = 82;
const Q_LEN: usize = 22;
const A_AT: usize = 104;
const B_AT: usize = 116;
const E: u32 = 174;

fn signatures(count: usize) -> Vec<([u8; 84], Vec<u8>, [u8; 142])> {
    let params = level1();
    let mut rng = ShakeRng::new(b"compact checks", b"tst");
    let (pk, sk) = compact_keygen::<P324_3, N>(&params, &mut rng).expect("keygen");
    let mut pkb = [0u8; 84];
    compact_public_key_to_bytes(&pk, &mut pkb).unwrap();
    (0..count)
        .map(|i| {
            let msg = format!("compact checks {}", i).into_bytes();
            let mut sig = [0u8; 142];
            compact_sign::<P324_3, N>(&params, &pk, &sk, &msg, &mut rng, &mut sig).expect("sign");
            (pkb, msg, sig)
        })
        .collect()
}

#[test]
fn rejects_a_and_b_both_even() {
    for (pkb, msg, sig) in signatures(4) {
        assert_eq!(hd_verify_bytes::<P324_3>(&sig, &pkb, &msg), Ok(()));
        let mut t = sig;
        t[A_AT] &= !1;
        t[B_AT] &= !1;
        assert_eq!(
            hd_verify_bytes::<P324_3>(&t, &pkb, &msg),
            Err(HdReject::ResponseRecovery),
            "a and b both even must be rejected by the recovery rule"
        );
    }
}

#[test]
fn rejects_a_degree_that_is_not_good() {
    for (pkb, msg, sig) in signatures(4) {
        assert_eq!(hd_verify_bytes::<P324_3>(&sig, &pkb, &msg), Ok(()));
        let q = Degree::from_le_bytes(&sig[Q_AT..Q_AT + Q_LEN]);
        assert!(is_good_degree(E, &q), "an honest degree is good");
        // q + 4k keeps q = 3 (mod 4), so 2^e - q stays 1 (mod 4); take the
        // first k for which 2^e - q - 4k is composite.
        let bad = (1u64..)
            .map(|k| q.wrapping_add(&Degree::from_u64(4 * k)))
            .find(|q2| !is_good_degree(E, q2))
            .unwrap();
        let mut t = sig;
        t[Q_AT..Q_AT + Q_LEN].copy_from_slice(&bad.to_le_bytes()[..Q_LEN]);
        assert_eq!(
            hd_verify_bytes::<P324_3>(&t, &pkb, &msg),
            Err(HdReject::DegreeNotGood),
            "a degree that is not good must be rejected by the goodness test"
        );
    }
}
