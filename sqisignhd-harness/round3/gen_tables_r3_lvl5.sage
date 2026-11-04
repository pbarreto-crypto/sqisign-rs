# Non-residue tables for the round-3 level V prime, the SQIsignHD library's rule
# (Signature/scripts/precompute_gf_constants.sage: seed 0, 20 entries, Fp2 = Fp[i]/(i^2+1)).
import sys
proof.all(False)
p = 17*2**664 - 1
out = sys.argv[1]
set_random_seed(0)
Fp2.<i> = GF((p,2), modulus=[1,0,1])
nqr = []
while len(nqr) < 20:
    e = Fp2.random_element()
    if not e.is_square():
        nqr.append(e)
znqr = []
while len(znqr) < 20:
    e = Fp2.random_element()
    if e.is_square() and not (e-1).is_square():
        znqr.append(e)
width = (p.nbits() + 7)//8*2
def line(e):
    a, b = e.polynomial().list() + [0]*(2 - len(e.polynomial().list()))
    return "0x{:0{w}x} + i*0x{:0{w}x}".format(int(a), int(b), w=width)
open(out + "/NQR_TABLE_lvl5.txt", "w").write("\n".join(line(e) for e in nqr) + "\n")
open(out + "/Z_NQR_TABLE_lvl5.txt", "w").write("\n".join(line(e) for e in znqr) + "\n")
print("p bits", p.nbits(), "width", width, "written")
