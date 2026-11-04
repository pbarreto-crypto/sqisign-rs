# Non-residue tables for the compact format's basis-from-hint convention, for a
# given prime p = c * 2^f - 1 (p = 3 mod 4, F_p^2 = F_p[i]/(i^2 + 1)).
# Deterministic rule: candidates a + b i, with a and b read from SHAKE-256
# (label "SQIsign compact NQR <name>" / "SQIsign compact ZNQR <name>"), each as
# ceil(bits(p)/8) + 16 bytes little-endian reduced mod p; the first twenty with
# the property are kept. In F_p^2, x is a square iff its norm a^2 + b^2 is a
# square in F_p.
import hashlib, sys
def legendre(x, p): return pow(x % p, (p - 1) // 2, p)
def is_sq(a, b, p): return legendre(a*a + b*b, p) in (0, 1)
def table(p, label, want):
    nb = (p.bit_length() + 7) // 8 + 16
    sh = hashlib.shake_256(label.encode()).digest(200000)
    out, pos = [], 0
    while len(out) < 20:
        a = int.from_bytes(sh[pos:pos+nb], 'little') % p; pos += nb
        b = int.from_bytes(sh[pos:pos+nb], 'little') % p; pos += nb
        if want(a, b): out.append((a, b))
    return out
def gen(name, p):
    nqr = table(p, f"SQIsign compact NQR {name}", lambda a, b: not is_sq(a, b, p))
    znqr = table(p, f"SQIsign compact ZNQR {name}", lambda a, b: is_sq(a, b, p) and not is_sq(a - 1, b, p))
    return nqr, znqr
def rust(name, p, nqr, znqr, header):
    w = (p.bit_length() + 7) // 8
    def enc(a, b): return list(a.to_bytes(w, 'little') + b.to_bytes(w, 'little'))
    def arr(n, rows):
        s = f"pub(crate) const {n}: [[u8; {2*w}]; 20] = [\n"
        for a, b in rows:
            bs = enc(a, b)
            s += "    [\n" + "".join("        " + ", ".join(str(x) for x in bs[i:i+20]) + ",\n" for i in range(0, len(bs), 20)) + "    ],\n"
        return s + "];\n"
    up = name.upper()
    return header + "\n/// Twenty quadratic non-residues, canonical encodings.\n" + arr(f"NQR_TABLE_{up}", nqr) + \
        "\n/// Twenty squares `x` with `x − 1` a non-residue, canonical encodings.\n" + arr(f"Z_NQR_TABLE_{up}", znqr)
if __name__ == "__main__":
    name, c, f = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    p = c * 2**f - 1
    w = (p.bit_length() + 7) // 8
    nqr, znqr = gen(name, p)
    header = f"""//! Non-residue tables for the round-3 prime `p = {c}·2^{f} − 1`, for the compact
//! format's basis-from-hint convention: `NQR_TABLE` holds quadratic
//! non-residues of `F_p^2`, `Z_NQR_TABLE` squares `x` with `x − 1` a
//! non-residue. Generated deterministically by `scripts/gen_nqr_tables.py`
//! (SHAKE-256 candidates; see the script), not by the SQIsignHD library's
//! seeded Sage rule used at level I: signer and verifier only need the same
//! tables. Each entry is the canonical little-endian `F_p^2` encoding
//! (re || im, {w} + {w} bytes).
"""
    open(f"nqr_tables_{name}.rs", "w").write(rust(name, p, nqr, znqr, header))
    # independent check of every entry
    assert all(not is_sq(a, b, p) for a, b in nqr)
    assert all(is_sq(a, b, p) and not is_sq(a - 1, b, p) for a, b in znqr)
    assert all(a < p and b < p for a, b in nqr + znqr)
    print(name, "p bits", p.bit_length(), "bytes per Fp", w, "tables written and checked")
