#![cfg(feature = "compact")]
//! The compact format at level V (`p = 17·2^664 − 1`, `e = 346`, `r = 175`):
//! sizes, round trips through the four-point and both two-point verifiers,
//! the fallback of the checked two-point verifier on honest signatures, and
//! rejection of tampered signatures in every mode.

use sqisign_rs::compact::{
    compact_keygen, compact_public_key_bytes, compact_public_key_to_bytes, compact_sign,
    compact_signature_bytes,
};
use sqisign_rs::mp::ShakeRng;
use sqisign_rs::sqisign::level5;
use sqisign_verify::hd::{hd_verify_bytes_with, HdVerifyOptions, KernelPoints};
use sqisign_verify::P664_17;

const N: usize = sqisign_rs::precomp::p664_17::IBZ_NLIMBS;
const SIG: usize = 280;
const PK: usize = 170;
const MODES: [KernelPoints; 3] = [
    KernelPoints::Four,
    KernelPoints::TwoWithImageCheck,
    KernelPoints::TwoWithoutImageCheck,
];

#[test]
fn sizes_level5() {
    assert_eq!(compact_signature_bytes::<P664_17>(), SIG);
    assert_eq!(compact_public_key_bytes::<P664_17>(), PK);
}

#[test]
fn sign_verify_tamper_level5() {
    let params = level5();
    let mut rng = ShakeRng::new(b"compact level V", b"tst");
    let (pk, sk) = compact_keygen::<P664_17, N>(&params, &mut rng).expect("keygen");
    let mut pkb = [0u8; PK];
    assert_eq!(compact_public_key_to_bytes(&pk, &mut pkb).unwrap(), PK);
    let (mut accepted, mut fell, mut rejected) = (0, 0, 0);
    for i in 0..6 {
        let msg = format!("level V message {i}").into_bytes();
        let mut s = [0u8; SIG];
        let n = compact_sign::<P664_17, N>(&params, &pk, &sk, &msg, &mut rng, &mut s).expect("sign");
        assert_eq!(n, SIG);
        for mode in MODES {
            let opts = HdVerifyOptions { kernel_points: mode };
            let (r, rep) = hd_verify_bytes_with::<P664_17>(&s, &pkb, &msg, &opts);
            assert!(r.is_ok(), "message {i}, mode {mode:?}: {r:?}");
            accepted += 1;
            fell += rep.fell_back.iter().filter(|&&b| b).count();
        }
        // tamper: one byte in each field of the signature
        for pos in [3usize, 170, 214, 236, 258, 278] {
            let mut t = s;
            t[pos] ^= 0x21;
            for mode in MODES {
                let opts = HdVerifyOptions { kernel_points: mode };
                let (r, _) = hd_verify_bytes_with::<P664_17>(&t, &pkb, &msg, &opts);
                assert!(r.is_err(), "tampered byte {pos} accepted, mode {mode:?}");
                rejected += 1;
            }
        }
        // the wrong message
        let (r, _) = hd_verify_bytes_with::<P664_17>(&s, &pkb, b"another message", &HdVerifyOptions { kernel_points: KernelPoints::TwoWithImageCheck });
        assert!(r.is_err());
    }
    assert_eq!(fell, 0, "the checked two-point verifier fell back on honest signatures");
    println!("LEVEL V: {accepted} verifications accepted (6 signatures x 3 modes), {rejected} tampered variants rejected, {fell} fallbacks");
}
