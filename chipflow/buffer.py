"""Back end: buffer insertion. One clustering routine serves three jobs:

- clock-tree synthesis: the flip-flop clock pins of each clock get a balanced tree of clock buffers;
- design-rule repair: nets driving more than `max_fanout` pins get a tree of buffers;
- hold fixing: a delay buffer goes in front of each flip-flop whose data arrives too early.

Trees are built bottom up: sinks are split by recursive median bisection along the wider spread
until each group has at most `cap` pins, a buffer drives each group from its centroid, and the
buffers' inputs are grouped the same way until the original driver sees at most `cap` loads.
New cells are appended to the netlist and placement, then the whole design is re-legalized.
"""
from __future__ import annotations

import numpy as np

from chipflow import place, sta


def _clusters(pts, idx, cap):
    if len(idx) <= cap:
        return [idx]
    p = pts[idx]
    axis = 0 if np.ptp(p[:, 0]) >= np.ptp(p[:, 1]) else 1
    o = idx[np.argsort(p[:, axis], kind="stable")]
    h = len(o) // 2
    return _clusters(pts, o[:h], cap) + _clusters(pts, o[h:], cap)


def _add_cell(net, pl, lef, name, typ, pins, cx, cy):
    m = lef["macros"][typ]
    net.cells.append((name, typ))
    net.pins.append(pins)
    pl.x = np.append(pl.x, cx - m.width / 2)
    pl.y = np.append(pl.y, cy - m.height / 2)
    pl.w = np.append(pl.w, m.width)
    pl.h = np.append(pl.h, m.height)
    pl.flip = np.append(pl.flip, False)
    return len(net.cells) - 1


def _new_net(net):
    net.n_nets += 1
    return net.n_nets - 1


def buffer_tree(net, lib, lef, pl, n, buf, cap, prefix) -> int:
    """Rebuild net n so that no net in the tree drives more than `cap` cell pins. Returns buffers added."""
    macros = lef["macros"]
    nodes = [(c, p) for c, pins in enumerate(net.pins) for p, k in pins.items()
             if k == n and lib[net.cells[c][1]].pins[p].direction != "output"]
    added = 0
    while len(nodes) > cap:
        pts = np.array([pl.pin_xy(c, p, macros, net.cells[c][1]) for c, p in nodes])
        nxt = []
        for grp in _clusters(pts, np.arange(len(nodes)), cap):
            out = _new_net(net)
            for i in grp:
                c, p = nodes[i]
                net.pins[c][p] = out
            cx, cy = pts[grp].mean(0)
            b = _add_cell(net, pl, lef, f"{prefix}{len(net.cells)}", buf, {"A": n, "Z": out}, cx, cy)
            nxt.append((b, "A"))
            added += 1
        nodes = nxt
    return added


def _legalize(pl, lef):
    centers = np.stack([pl.x + pl.w / 2, pl.y + pl.h / 2], 1)
    pl.x, pl.y, pl.flip = place.legalize(pl.core, lef, pl.w, pl.h, centers)


def cts(net, lib, lef, pl, cap=16, buf="CLKBUF_X3") -> dict:
    """Clock-tree synthesis: one balanced buffer tree per clock port."""
    drv, _ = net.drivers_sinks(lib)
    _, clk_nets, roots = sta.clock_network(net, lib, drv)
    added = 0
    for n in sorted(clk_nets):
        if drv.get(n, ("",))[0] == "port":
            added += buffer_tree(net, lib, lef, pl, n, buf, cap, "cts_")
    _legalize(pl, lef)
    return {"clock_ports": sorted(roots), "buffers": added, "max_fanout": cap, "cell": buf}


def repair_design(net, lib, lef, pl, period_ns, max_fanout=16, buf="BUF_X4", size_rounds=3) -> dict:
    """Buffer every cell-driven, non-clock net whose fanout exceeds max_fanout, then upsize the
    drivers of nets that still break the slew or capacitance limits (weak driver, long wire)."""
    from chipflow.size import _bigger
    macros = lef["macros"]
    t = sta.analyze(net, lib, period_ns, pl, macros, hold=False)
    before = {"slew": t["slew_violations"], "cap": t["cap_violations"]}
    drv, sinks = net.drivers_sinks(lib)
    _, clk_nets, _ = sta.clock_network(net, lib, drv)
    targets = [n for n, d in drv.items() if d[0] != "port" and n not in clk_nets and n not in net.const
               and sum(s[0] != "port" for s in sinks[n]) > max_fanout]
    added = sum(buffer_tree(net, lib, lef, pl, n, buf, max_fanout, "buf_") for n in targets)
    upsized = 0
    for _ in range(size_rounds):
        t = sta.analyze(net, lib, period_ns, pl, macros, hold=False)
        drv, _ = net.drivers_sinks(lib)
        cells = {drv[n][0] for n in t["slow_nets"] | t["cap_nets"] if n in drv and drv[n][0] != "port"}
        bump = [c for c in cells if _bigger(lib, net.cells[c][1])]
        if not bump:
            break
        for c in bump:
            net.cells[c] = (net.cells[c][0], _bigger(lib, net.cells[c][1]))
            pl.w[c] = macros[net.cells[c][1]].width
        upsized += len(bump)
    _legalize(pl, lef)
    t = sta.analyze(net, lib, period_ns, pl, macros, hold=False)
    return {"nets_buffered": len(targets), "buffers": added, "drivers_upsized": upsized, "max_fanout": max_fanout,
            "cell": buf, "violations_before": before, "violations_after": {"slew": t["slew_violations"], "cap": t["cap_violations"]}}


def fix_hold(net, lib, lef, pl, period_ns, buf="BUF_X1", rounds=4, setup_margin=0.05, net_len=None) -> dict:
    """Delay buffers in front of flip-flops that fail hold, where their setup slack can afford it."""
    macros = lef["macros"]
    t = sta.analyze(net, lib, period_ns, pl, macros, net_len)
    start, added = t["whs_ns"], 0
    for _ in range(rounds):
        bad = [c for c, s in t["hold_by_ff"].items() if s < 0]
        if not bad:
            break
        for c in bad:
            d = lib[net.cells[c][1]].ff["next"]
            n_d = net.pins[c][d]
            if t["net_slack"].get(n_d, 0.0) < setup_margin:
                continue  # no setup room for more delay on this path
            out = _new_net(net)
            x, y = pl.pin_xy(c, d, macros, net.cells[c][1])
            _add_cell(net, pl, lef, f"hold_{len(net.cells)}", buf, {"A": n_d, "Z": out}, x, y)
            net.pins[c][d] = out
            added += 1
        _legalize(pl, lef)
        t = sta.analyze(net, lib, period_ns, pl, macros, net_len)
    return {"buffers": added, "whs_before_ns": start, "whs_after_ns": t["whs_ns"], "hold_failing": t["hold_failing"]}
