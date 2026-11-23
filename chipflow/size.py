"""Back end, timing repair: gate sizing with the library's drive strengths (X1 -> X2 -> X4 ...).

Greedy, slack-driven: each round upsizes the worst-slack cells one step, in a batch. A batch is
kept only if WNS improves; otherwise it is undone and a smaller batch is tried, because a bigger
gate also loads its own driver harder. Wider cells are re-legalized afterwards.
"""
from __future__ import annotations

import re

import numpy as np

from chipflow import place, sta


def _bigger(lib, typ):
    m = re.fullmatch(r"(.+)_X(\d+)", typ)
    if not m:
        return None
    sizes = sorted(int(re.fullmatch(r".+_X(\d+)", t).group(1)) for t in lib if re.fullmatch(re.escape(m.group(1)) + r"_X\d+", t))
    up = [s for s in sizes if s > int(m.group(2))]
    return f"{m.group(1)}_X{up[0]}" if up else None


def upsize(net, lib, lef, pl, period_ns, max_rounds=40, log=None, net_len=None) -> dict:
    """net_len: routed lengths for post-route sizing (else placed HPWL estimates)."""
    macros = lef["macros"]
    area = lambda: float(sum(lib[t].area for _, t in net.cells))
    t = sta.analyze(net, lib, period_ns, pl, macros, net_len)
    start = {"wns_ns": t["wns_ns"], "area_um2": round(area(), 1)}
    changed, batch_frac, rounds = set(), 0.25, 0
    while rounds < max_rounds and t["wns_ns"] < 0:
        rounds += 1
        ns = t["net_slack"]
        crit = []
        for c, (inst, typ) in enumerate(net.cells):
            outs = [net.pins[c][p.name] for p in lib[typ].outputs() if p.name in net.pins[c]]
            s = min((ns.get(n, np.inf) for n in outs), default=np.inf)
            if s < 0 and _bigger(lib, typ):
                crit.append((s, c))
        if not crit:
            break
        crit.sort()
        batch = [c for _, c in crit[:max(1, int(len(crit) * batch_frac))]]
        old = {c: net.cells[c] for c in batch}
        for c in batch:
            inst, typ = net.cells[c]
            net.cells[c] = (inst, _bigger(lib, typ))
            pl.w[c] = macros[net.cells[c][1]].width
        t2 = sta.analyze(net, lib, period_ns, pl, macros, net_len)
        if t2["wns_ns"] > t["wns_ns"] + 1e-4:
            t = t2
            changed.update(batch)
            if log:
                log(f"  size round {rounds:2d}: upsized {len(batch):4d} cells, WNS {t['wns_ns']:+.4f} ns")
        else:
            for c, v in old.items():
                net.cells[c] = v
                pl.w[c] = macros[v[1]].width
            if len(batch) == 1:
                break
            batch_frac /= 2
    centers = np.stack([pl.x + pl.w / 2, pl.y + pl.h / 2], 1)
    pl.x, pl.y, pl.flip = place.legalize(pl.core, lef, pl.w, pl.h, centers)
    final = sta.analyze(net, lib, period_ns, pl, macros, net_len)
    return {"rounds": rounds, "cells_upsized": len(changed), "wns_before_ns": start["wns_ns"],
            "wns_after_ns": final["wns_ns"], "area_before_um2": start["area_um2"], "area_after_um2": round(area(), 1)}
