"""Built-in designs and their golden-model testbenches (front-end verification + switching activity)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from chipflow.sim import pack, unpack

DESIGN_DIR = Path(__file__).resolve().parent.parent / "designs"


def systolic_tb(sim, params: dict, rng, lanes: int = 64) -> dict:
    """Random signed A (N x K) times B (K x N) per lane, fed with the classic skew:
    row i of A enters i cycles late, column j of B j cycles late. Checks every PE."""
    n, w, acc = params["N"], params.get("W", 8), params.get("ACC", 24)
    k = n
    a = rng.integers(-(1 << (w - 1)), 1 << (w - 1), size=(lanes, n, k))
    b = rng.integers(-(1 << (w - 1)), 1 << (w - 1), size=(lanes, k, n))
    zero = np.zeros(n * w, dtype=np.uint64)
    sim.step({"clr": np.array([np.uint64(0xFFFFFFFFFFFFFFFF)]), "a_west": zero, "b_north": zero})
    mask = (1 << w) - 1
    for s in range(k + 2 * n - 1 + params.get("REG_IN", 0)):  # input registers add a cycle
        west = np.concatenate([pack([a[l, i, s - i] & mask if 0 <= s - i < k else 0 for l in range(lanes)], w)
                               for i in range(n)])
        north = np.concatenate([pack([b[l, s - j, j] & mask if 0 <= s - j < k else 0 for l in range(lanes)], w)
                                for j in range(n)])
        out = sim.step({"clr": np.zeros(1, dtype=np.uint64), "a_west": west, "b_north": north})
    want = np.einsum("lik,lkj->lij", a, b)
    got = np.array([unpack(out["acc"][(p * acc):(p + 1) * acc], lanes, signed=True) for p in range(n * n)]).T
    errors = int((got != want.reshape(lanes, n * n)).sum())
    return {"checked": lanes * n * n, "mismatches": errors, "cycles": sim.cycles}


def simt_ref(op, a, b, c, imm, lane, w):
    """Golden semantics of one simt_core lane (mirror of the case statement in simt_core.v)."""
    m = (1 << w) - 1
    s = lambda x: x - (1 << w) if x >> (w - 1) else x
    sh = b & (w - 1)
    lt = s(a) < s(b)
    return {1: a + b, 2: a - b, 3: a & b, 4: a | b, 5: a ^ b, 6: a << sh, 7: a >> sh, 8: a * b, 9: int(lt),
            10: imm - 256 if imm & 128 else imm, 11: lane, 12: c + a * b, 13: a if lt else b,
            14: b if lt else a}.get(op, c) & m


def simt_tb(sim, params: dict, rng, lanes: int = 64, cycles: int = 200) -> dict:
    """Random instruction streams, warps and divergence masks; each sim lane is an independent
    program. Compares every lane's registered write-back with the golden ISA model each cycle."""
    nl, nw, nr, w = params.get("LANES", 4), params.get("WARPS", 2), params.get("REGS", 8), params.get("W", 16)
    regs = np.zeros((lanes, nw, nr, nl), dtype=np.int64)  # sim lane, warp, reg, hw lane
    wid_bits = max(1, (nw - 1).bit_length())
    hist, checked, errors = [], 0, 0
    delay = 1 + params.get("REG_IN", 0)  # write-back is registered; an issue register adds a cycle
    for t in range(cycles):
        boot = t < nw * nr                     # first: LI every register so state is defined
        op = np.full(lanes, 10) if boot else rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0], size=lanes)
        rd = np.full(lanes, t % nr) if boot else rng.integers(0, nr, lanes)
        rs1, rs2 = rng.integers(0, nr, lanes), rng.integers(0, nr, lanes)
        imm = rng.integers(0, 256, lanes)
        wid = np.full(lanes, (t // nr) % nw) if boot else rng.integers(0, nw, lanes)
        mask = np.full(lanes, (1 << nl) - 1) if boot else rng.integers(0, 1 << nl, lanes)
        valid = np.ones(lanes, int) if boot else (rng.random(lanes) < 0.9).astype(int)
        instr = (op << 12) | (rd << 9) | np.where(op == 10, imm, (rs1 << 6) | (rs2 << 3))
        rs1 = np.where(op == 10, (imm >> 6) & 7, rs1)  # LI's imm overlaps rs1/rs2 fields
        rs2 = np.where(op == 10, (imm >> 3) & 7, rs2)
        out = sim.step({"clk": np.zeros(1, np.uint64), "valid": pack(valid, 1), "wid": pack(wid, wid_bits),
                        "instr": pack(instr, 16), "mask": pack(mask, nl)})
        if len(hist) >= delay:
            got = np.array([unpack(out["wb"][h * w:(h + 1) * w], lanes) for h in range(nl)]).T
            checked += got.size
            errors += int((got != hist[-delay]).sum())
        want = np.zeros((lanes, nl), dtype=np.int64)
        for s_ in range(lanes):
            for h in range(nl):
                rf = regs[s_, wid[s_], :, h]
                r = simt_ref(int(op[s_]), int(rf[rs1[s_]]), int(rf[rs2[s_]]), int(rf[rd[s_]]), int(imm[s_]), h, w)
                we = valid[s_] and (mask[s_] >> h) & 1
                want[s_, h] = r if we else 0
                if we and op[s_] not in (0, 15):
                    regs[s_, wid[s_], rd[s_], h] = r
        hist.append(want)
    return {"checked": checked, "mismatches": errors, "cycles": sim.cycles}


DESIGNS = {
    "systolic": {"sources": [DESIGN_DIR / "systolic_array.v"], "top": "systolic_array",
                 "params": {"N": 4, "W": 8, "ACC": 24, "REG_IN": 1}, "tb": systolic_tb},
    "simt": {"sources": [DESIGN_DIR / "simt_core.v"], "top": "simt_core",
             "params": {"LANES": 4, "WARPS": 2, "REGS": 8, "W": 16, "REG_IN": 1}, "tb": simt_tb},
}
