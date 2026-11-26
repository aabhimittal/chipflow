"""Levelized, vectorized static timing: the model of chipflow.sta, evaluated one logic level at a
time with numpy instead of one arc at a time in Python.

Cells are grouped into levels (every input driven by a lower level), so a level is a handful of
array operations: gather input arrivals, look up every arc delay at once in the packed 7x7 NLDM
tables, keep the latest (or earliest) candidate per output net and edge. Ties go to the first
candidate in the Python engine's order (arc, then input edge), so the two engines agree bit for
bit. Path tags become lanes of the arrays; this needs every tag known at the start points, so
designs with -through exceptions use the Python engine (chipflow.sta.analyze dispatches).
"""
from __future__ import annotations

import numpy as np

from chipflow import pdk
from chipflow.sta import (CK, CLK_SLEW, IN_SLEW, MAX_SLEW, OUT_LOAD_FF, RF, RISE, UNTIMED, _context, _template)

_PACKS: dict = {}


def _pack(lib):
    """All combinational delay/slew tables of a library as arrays: X, Y (n, 7) and V (n, 7, 7)."""
    if id(lib) not in _PACKS:
        tabs = {}
        for cell in lib.values():
            for _, arcs in _template(cell):
                for _, edges in arcs:
                    for dtab, stab, _ in edges.values():
                        for t in (dtab, stab):
                            tabs.setdefault(id(t), t)
        ts = list(tabs.values())
        assert all(len(t.x) == 7 and len(t.y) == 7 for t in ts), "vector engine expects 7x7 tables"
        _PACKS[id(lib)] = (lib, {id(t): i for i, t in enumerate(ts)}, np.array([t.x for t in ts]),
                           np.array([t.y for t in ts]), np.array([t.v for t in ts]))
    return _PACKS[id(lib)][1:]


def _interp(X, Y, V, tid, a, b):
    """Table.__call__ on arrays, same operations in the same order (bit-identical)."""
    r = np.arange(len(tid))
    x, y = X[tid], Y[tid]
    i = np.clip((x <= a[:, None]).sum(1) - 1, 0, 5)
    j = np.clip((y <= b[:, None]).sum(1) - 1, 0, 5)
    tx = (a - x[r, i]) / (x[r, i + 1] - x[r, i])
    ty = (b - y[r, j]) / (y[r, j + 1] - y[r, j])
    r0j, r0j1, r1j, r1j1 = V[tid, i, j], V[tid, i, j + 1], V[tid, i + 1, j], V[tid, i + 1, j + 1]
    v0 = r0j + ty * (r0j1 - r0j)
    return v0 + tx * (r1j + ty * (r1j1 - r1j) - v0)


def analyze(net, lib, period_ns: float, pl=None, macros=None, net_len=None, hold=True, graph=False) -> dict:
    """graph=True adds the levelized graph and operating point under "_graph" (for chipflow.place)."""
    tab_id, X, Y, V = _pack(lib)
    drv, sinks = net.drivers_sinks(lib)
    cx = _context(net, lib, period_ns, drv)
    assert not cx.thr_of, "-through exceptions need the Python engine"
    io, N = net.io, net.n_nets
    r_um, c_um = pdk.WIRE_R_OHM_UM, pdk.WIRE_C_FF_UM
    out_load = (lambda name: io["output_load"].get(name, 0.0)) if io else (lambda name: OUT_LOAD_FF)
    unc_setup, unc_hold = io["uncertainty"] if io else (0.0, 0.0)

    # --- loads and wires: one record per sink pin, in the order the Python engine sums them ---
    offs, caps = {}, {}

    def off(typ, pin):
        if (typ, pin) not in offs:
            m = macros[typ].pins.get(pin) if macros else None
            offs[typ, pin] = (m[0], m[1], 1) if m else (0.0, 0.0, 0)
        return offs[typ, pin]

    recs = [(n, p) for n in range(N) for p in sinks[n]]          # every sink pin, in each net's pin order
    rec_of = dict(zip(recs, range(len(recs))))
    keys = [("port", p[1]) if p[0] == "port" else (net.cells[p[0]][1], p[1]) for _, p in recs]
    kid, klist = {}, []
    for k in keys:
        if k not in kid:
            kid[k] = len(klist)
            klist.append(k)
    kcap = np.array([out_load(k[1]) if k[0] == "port" else lib[k[0]].pins[k[1]].cap for k in klist])
    koff = np.array([off(*k) if k[0] != "port" else (0.0, 0.0, 0) for k in klist]).reshape(-1, 3)
    ki = np.array([kid[k] for k in keys], int)
    rn = np.array([n for n, _ in recs], int)
    rcap = kcap[ki] if len(ki) else np.zeros(0)
    rc = np.array([p[0] if p[0] != "port" else -1 for _, p in recs], int)
    ports_xy = np.array([(pl.io[(p[1], p[2])] if pl is not None else (0.0, 0.0)) if p[0] == "port" else (0.0, 0.0) for _, p in recs]).reshape(-1, 2)
    rdx, rdy, rho = (koff[ki, 0], koff[ki, 1], koff[ki, 2]) if len(ki) else (np.zeros(0),) * 3
    rpx, rpy = ports_xy[:, 0], ports_xy[:, 1]

    def xy(c, dx, dy, h, px, py):
        c, dx, dy, h = np.array(c, int), np.array(dx), np.array(dy), np.array(h, bool)
        if pl is None:
            return np.zeros(len(c)), np.zeros(len(c))
        cc = np.maximum(c, 0)
        w, hh = pl.w[cc], pl.h[cc]
        dx, dy = np.where(h, dx, w / 2), np.where(h, dy, hh / 2)
        x = np.where(c >= 0, pl.x[cc] + dx, np.array(px))
        y = np.where(c >= 0, pl.y[cc] + np.where(pl.flip[cc], hh - dy, dy), np.array(py))
        return x, y

    sx, sy = xy(rc, rdx, rdy, rho, rpx, rpy)
    def point(p):  # (cell or -1, dx, dy, has offset, port x, port y)
        if p[0] == "port":
            x, y = pl.io[(p[1], p[2])] if pl is not None else (0.0, 0.0)
            return -1, 0.0, 0.0, 0, x, y
        dx, dy, h = off(net.cells[p[0]][1], p[1])
        return p[0], dx, dy, h, 0.0, 0.0

    has_drv = np.zeros(N, bool)
    dpts = [(0, 0.0, 0.0, 0, 0.0, 0.0)] * N
    for n, d in drv.items():
        has_drv[n] = True
        dpts[n] = point(d)
    dx0, dy0 = xy(*zip(*dpts)) if N else (np.zeros(0), np.zeros(0))
    # Python's sum() is compensated (Neumaier) since 3.12, so it is used per net to match exactly
    cut = np.searchsorted(rn, np.arange(N + 1))
    caps_l = rcap.tolist()
    cpin = np.array([sum(caps_l[cut[n]:cut[n + 1]]) for n in range(N)])
    timed = has_drv[rn] if pl is not None else np.zeros(len(rn), bool)   # wires only where Python has them
    length = np.zeros(N)
    if pl is not None:
        mxx, mnx, mxy, mny = dx0.copy(), dx0.copy(), dy0.copy(), dy0.copy()
        np.maximum.at(mxx, rn, sx), np.minimum.at(mnx, rn, sx), np.maximum.at(mxy, rn, sy), np.minimum.at(mny, rn, sy)
        hp = mxx - mnx + mxy - mny
        has_pins = np.bincount(rn, minlength=N) > 0
        length = np.where(has_drv & has_pins, hp, 0.0)
        if net_len:
            for n, v in net_len.items():
                if has_drv[n] and has_pins[n]:
                    length[n] = v
    load = cpin + c_um * length
    d = np.abs(sx - dx0[rn]) + np.abs(sy - dy0[rn])
    wire = np.where(timed, r_um * d * (c_um * d / 2 + rcap) * 1e-6, 0.0)
    wget = lambda n, p: float(wire[rec_of[(n, p)]]) if (n, p) in rec_of else 0.0

    def sink_wire(cs, typ, pin, nets):
        """Wire delay into pin `pin` of cells cs (all of type typ) from the drivers of `nets`."""
        if pl is None:
            return np.zeros(len(cs))
        dx, dy, h = off(typ, pin)
        x, y = xy(cs, np.full(len(cs), dx), np.full(len(cs), dy), np.full(len(cs), h), np.zeros(len(cs)), np.zeros(len(cs)))
        cp = lib[typ].pins[pin].cap
        d = np.abs(x - dx0[nets]) + np.abs(y - dy0[nets])
        return np.where(has_drv[nets], r_um * d * (c_um * d / 2 + cp) * 1e-6, 0.0)

    # --- candidates: one row per (cell, output, arc, output edge, input edge), grouped by cell type ---
    clk = cx.clk_cells
    comb_drv = {}
    for c in cx.order:
        for pin, _ in _template(lib[net.cells[c][1]]):
            if pin in net.pins[c]:
                comb_drv[net.pins[c][pin]] = c
    level = {}
    for c in cx.order:
        lv = 0
        for p, n in net.pins[c].items():
            dc = comb_drv.get(n)
            if dc is not None and dc != c:
                lv = max(lv, level[dc] + 1)
        level[c] = lv
    by_type = {}
    for c in cx.order:
        by_type.setdefault(net.cells[c][1], []).append(c)
    cols = {k: [] for k in ("cell", "lev", "clk", "nout", "tout", "nin", "tin", "rank", "dtab", "stab", "rel", "w")}
    rel_names = []
    rel_id = {}
    for typ, cs in by_type.items():
        cs = np.array(cs)
        pins = {}
        rows, rank = [], 0
        for opin, arcs in _template(lib[typ]):
            for related, edges in arcs:
                for t_out, (dtab, stab, ins) in edges.items():
                    for t_in, _ in ins:
                        rows.append((opin, related, RF.index(t_out), RF.index(t_in), tab_id[id(dtab)], tab_id[id(stab)], rank))
                        rank += 1
        for opin, related, *_ in rows:
            for pn in (opin, related):
                if pn not in pins:
                    pins[pn] = np.array([net.pins[c].get(pn, -1) for c in cs])
            if related not in rel_id:
                rel_id[related] = len(rel_names)
                rel_names.append(related)
        lev = np.array([level[c] for c in cs])
        isclk = np.array([c in clk for c in cs])
        for opin, related, to, ti, dt, st, rk in rows:
            no, ni = pins[opin], pins[related]
            ok = (no >= 0) & (ni >= 0)
            if not ok.any():
                continue
            k = int(ok.sum())
            cols["cell"].append(cs[ok]), cols["lev"].append(lev[ok]), cols["clk"].append(isclk[ok])
            cols["nout"].append(no[ok]), cols["nin"].append(ni[ok])
            for key, v in (("tout", to), ("tin", ti), ("rank", rk), ("dtab", dt), ("stab", st), ("rel", rel_id[related])):
                cols[key].append(np.full(k, v))
            cols["w"].append(sink_wire(cs[ok], typ, related, ni[ok]))
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    # clock cells first, then by level, output net, output edge and the Python iteration rank
    o = np.lexsort((cat["rank"], cat["tout"], cat["nout"], cat["lev"], ~cat["clk"]))
    ccell, lev, isclk = cat["cell"][o].astype(int), cat["lev"][o].astype(int), cat["clk"][o].astype(bool)
    nout, tout, nin, tin = (cat[k][o].astype(int) for k in ("nout", "tout", "nin", "tin"))
    dtab, stab, rel, w = cat["dtab"][o].astype(int), cat["stab"][o].astype(int), cat["rel"][o].astype(int), cat["w"][o]
    K = len(ccell)
    # level slices (clock part, then data part) and the start of every (net, edge) group inside each
    seg = np.flatnonzero(np.r_[True, (lev[1:] != lev[:-1]) | (isclk[1:] != isclk[:-1])]) if K else np.zeros(0, int)
    bounds = list(zip(seg, list(seg[1:]) + [K]))
    grp = np.r_[True, (nout[1:] != nout[:-1]) | (tout[1:] != tout[:-1]) | (lev[1:] != lev[:-1]) | (isclk[1:] != isclk[:-1])] if K else np.zeros(0, bool)
    levels = [(lo, hi, np.flatnonzero(grp[lo:hi]), bool(isclk[lo])) for lo, hi in bounds]

    # --- start points: every tag must be known now (the lanes of the arrays) ---
    ffs = cx.ffs
    primary = cx.primary
    arc_of = cx.arc_of

    def sources(early):
        """[(net, lane, edge, arrival, slew)] for inputs and constants."""
        out = []
        for name, (dd, nets) in net.ports.items():
            if dd != "input":
                continue
            for n in nets:
                if name in cx.clk_roots:
                    out += [(n, CK, t, 0.0, IN_SLEW) for t in (0, 1)]
                elif not io:
                    g = UNTIMED if early else cx.start_tag(primary, ("port", name), n)
                    out += [(n, g, t, 0.0, IN_SLEW) for t in (0, 1)]
                elif name not in io["input_delay"]:
                    out += [(n, UNTIMED, t, 0.0, IN_SLEW) for t in (0, 1)]
                else:
                    dl, arc = io["input_delay"][name][1 if early else 0], arc_of.get(io["driver"].get(name))
                    g = cx.start_tag(io.get("input_clock", {}).get(name, primary), ("port", name), n)
                    for t, e in enumerate(RF):
                        out.append((n, g, t, dl + (arc.tables[f"cell_{RISE[e]}"](IN_SLEW, load[n]) if arc else 0.0),
                                    arc.tables[f"{RISE[e]}_transition"](IN_SLEW, load[n]) if arc else IN_SLEW))
        for n in net.const:
            out += [(n, UNTIMED, t, 0.0, IN_SLEW) for t in (0, 1)]
        return out

    src = {False: sources(False), True: sources(True) if hold else []}
    launch_tag = {}
    for c in ffs:
        for p in lib[net.cells[c][1]].outputs():
            n = net.pins[c].get(p.name)
            if n is not None:
                launch_tag[c, p.name] = (n, cx.start_tag(cx.ff_clock[c], c, n))
    L = len(cx.tags)

    def forward(early):
        A = np.full((L, 2, N), np.inf if early else -np.inf)
        S = np.full((L, 2, N), IN_SLEW)
        P = np.full((L, 2, N), -1, dtype=np.int64)       # >=0 candidate row, -1 start point, <=-2 launch of FF -2-P
        D = np.zeros((L, K)) if not early else None
        for n, g, t, a, s in src[early]:
            A[g, t, n], S[g, t, n] = a, s
        red = np.minimum if early else np.maximum

        def run(clock_part):
            for lo, hi, gs, part in levels:
                if part != clock_part:
                    continue
                sl = slice(lo, hi)
                ni, ti, no, to = nin[sl], tin[sl], nout[sl], tout[sl]
                ld = load[no]
                for l in range(L):
                    a_in = A[l, ti, ni]
                    ok = np.isfinite(a_in)
                    if not ok.any():
                        continue
                    s_in = np.where(ok, S[l, ti, ni], IN_SLEW)
                    dly = _interp(X, Y, V, dtab[sl], s_in, ld)
                    if D is not None:
                        D[l, sl] = dly
                    cand = a_in + w[sl] + dly
                    best = red.reduceat(cand, gs)
                    cnt = np.diff(np.r_[gs, hi - lo])
                    first = np.minimum.reduceat(np.where(cand == np.repeat(best, cnt), np.arange(hi - lo), hi - lo), gs)
                    keep = np.isfinite(best)
                    f = first[keep]
                    A[l, to[gs][keep], no[gs][keep]] = best[keep]
                    S[l, to[gs][keep], no[gs][keep]] = _interp(X, Y, V, stab[sl][f], s_in[f], ld[f])
                    P[l, to[gs][keep], no[gs][keep]] = lo + f

        run(True)                                       # the clock tree first: flip-flops launch off it
        lat = {}
        for c in ffs:
            cell = lib[net.cells[c][1]]
            ck = cell.ff["clock"]
            n_ck = net.pins[c][ck]
            a = A[CK, 0, n_ck]
            lat[c] = (float(a) + wget(n_ck, (c, ck)), float(S[CK, 0, n_ck])) if cx.propagated and np.isfinite(a) else (0.0, CLK_SLEW)
            Lc, s_ck = lat[c]
            for p in cell.outputs():
                if (c, p.name) not in launch_tag:
                    continue
                n, g = launch_tag[c, p.name]
                arc = next(x for x in p.arcs if x.kind == "rising_edge")
                for t, e in enumerate(RF):
                    A[g, t, n] = Lc + arc.tables[f"cell_{RISE[e]}"](s_ck, load[n])
                    S[g, t, n] = arc.tables[f"{RISE[e]}_transition"](s_ck, load[n])
                    P[g, t, n] = -2 - c
        run(False)
        return A, S, P, D, lat

    A, S, P, D, lat = forward(False)
    tags, check, rel_fn, clocks = cx.tags, cx.check, cx.rel, cx.clocks
    present = lambda Arr, g, t, n: bool(np.isfinite(Arr[g, t, n]))
    lanes_at = lambda Arr, n: [g for g in range(L) if np.isfinite(Arr[g, :, n]).any()]

    ends = []
    for c in ffs:
        inst, typ = net.cells[c]
        cell = lib[typ]
        dpin = cell.ff["next"]
        arc = next((a for a in cell.pins[dpin].arcs if a.kind == "setup_rising"), None)
        n = net.pins[c][dpin]
        Lc, s_ck = lat[c]
        cap = cx.ff_clock[c]
        for g in lanes_at(A, n):
            chk = check(c, cap, g)
            if chk is None:
                continue
            R = rel_fn(tags[g][0], cap)[0] + (chk[0] - 1) * clocks[cap][0]
            for t, e in enumerate(RF):
                if present(A, g, t, n):
                    a, s = float(A[g, t, n]) + wget(n, (c, dpin)), float(S[g, t, n])
                    su = arc.tables[f"{RISE[e]}_constraint"](s, s_ck) if arc else 0.0
                    ends.append((R + Lc - su - a - unc_setup, f"{inst}/{dpin}", n, e, (c, dpin), R, g))
    for name, (dd, nets) in net.ports.items():
        if dd != "output" or (io and name not in io["output_delay"]):
            continue
        cap = io.get("output_clock", {}).get(name, primary) if io else primary
        od = io["output_delay"][name][0] if io else 0.0
        for b, n in enumerate(nets):
            for g in lanes_at(A, n):
                chk = check(("port", name), cap, g)
                if chk is None:
                    continue
                R = rel_fn(tags[g][0], cap)[0] + (chk[0] - 1) * clocks[cap][0]
                for t, e in enumerate(RF):
                    if present(A, g, t, n):
                        a = float(A[g, t, n]) + wget(n, ("port", name, b))
                        ends.append((R - od - a - unc_setup, f"{name}[{b}]", n, e, ("port", name, b), R, g))
    ends.sort(key=lambda e: e[0])

    # backward pass, level by level in reverse, with the forward pass's delays
    req = np.full((L, 2, N), np.inf)
    for sl_, _, n, e, _, _, g in ends:
        t = RF.index(e)
        req[g, t, n] = min(req[g, t, n], sl_ + A[g, t, n])
    for lo, hi, gs, part in reversed(levels):
        if part:
            continue
        sl = slice(lo, hi)
        for l in range(L):
            r_out = req[l, tout[sl], nout[sl]]
            ok = np.isfinite(r_out) & np.isfinite(A[l, tin[sl], nin[sl]])
            if ok.any():
                np.minimum.at(req[l], (tin[sl][ok], nin[sl][ok]), (r_out - D[l, sl] - w[sl])[ok])
    fin = np.isfinite(req)
    ns_val = np.where(fin, req - np.where(fin, A, 0.0), np.inf).min(axis=(0, 1))
    net_slack = {int(n): ns_val[n] for n in np.flatnonzero(fin.any(axis=(0, 1)))}

    holdl, port_hold = [], []
    if hold:
        Ae, Se, _, _, _ = forward(True)
        for c in ffs:
            inst, typ = net.cells[c]
            cell = lib[typ]
            dpin = cell.ff["next"]
            arc = next((a for a in cell.pins[dpin].arcs if a.kind == "hold_rising"), None)
            n = net.pins[c][dpin]
            Lc, s_ck = lat[c]
            cap = cx.ff_clock[c]
            for g in lanes_at(Ae, n):
                chk = check(c, cap, g)
                if chk is None:
                    continue
                H = rel_fn(tags[g][0], cap)[1] + (chk[0] - 1 - chk[1]) * clocks[cap][0]
                for t, e in enumerate(RF):
                    if present(Ae, g, t, n):
                        a, s = float(Ae[g, t, n]) + wget(n, (c, dpin)), float(Se[g, t, n])
                        h = arc.tables[f"{RISE[e]}_constraint"](s, s_ck) if arc else 0.0
                        holdl.append((a - Lc - h - H - unc_hold, f"{inst}/{dpin}", c))
        for name, (dd, nets) in (net.ports.items() if io else []):
            if dd != "output" or name not in io["output_delay"]:
                continue
            cap = io.get("output_clock", {}).get(name, primary)
            for b, n in enumerate(nets):
                for g in lanes_at(Ae, n):
                    chk = check(("port", name), cap, g)
                    if chk is None:
                        continue
                    H = rel_fn(tags[g][0], cap)[1] + (chk[0] - 1 - chk[1]) * clocks[cap][0]
                    a = min(float(Ae[g, t, n]) + wget(n, ("port", name, b)) for t in (0, 1) if present(Ae, g, t, n))
                    port_hold.append(a + io["output_delay"][name][1] - H - unc_hold)
    holdl.sort()
    hold_by_ff = {}
    for sl_, _, c in holdl:
        hold_by_ff.setdefault(c, sl_)

    drv_cap = {}
    for n, dd in drv.items():
        if dd[0] != "port":
            mc = lib[net.cells[dd[0]][1]].pins[dd[1]].max_cap
        elif io and dd[1] not in cx.clk_roots and dd[1] in io["driver"]:
            mc = next(p.max_cap for p in lib[io["driver"][dd[1]]].outputs())
        else:
            continue
        if load[n] > mc:
            drv_cap[n] = load[n]
    slowm = (np.isfinite(A) & (S > MAX_SLEW)).any(axis=(0, 1))
    slow = {int(n) for n in np.flatnonzero(slowm)} - set(net.const)
    lats = [lat[c][0] for c in ffs]

    def trace(end):
        _, _, n, e, sink, _, g = end
        t = RF.index(e)
        wd = wget(n, sink)
        stages = []
        while True:
            a, pr = float(A[g, t, n]), int(P[g, t, n])
            if pr == -1:
                stages.append({"point": "input", "arrival_ns": round(a, 4)})
                break
            if pr <= -2:
                c = -2 - pr
                pin = next(p for (cc, p), (nn, _) in launch_tag.items() if cc == c and nn == n)
                stages.append({"idx": c, "point": f"{net.cells[c][0]}/{pin}", "cell": net.cells[c][1], "edge": RF[t],
                               "incr_ns": round(a, 4), "wire_ns": round(wd, 4), "arrival_ns": round(a, 4)})
                break
            c = int(ccell[pr])
            stages.append({"idx": c, "point": f"{net.cells[c][0]}", "cell": net.cells[c][1], "edge": RF[t],
                           "incr_ns": round(float(D[g, pr]), 4), "wire_ns": round(wd, 4), "arrival_ns": round(a, 4)})
            n, t, wd = int(nin[pr]), int(tin[pr]), float(w[pr])
        return stages[::-1]

    wns = ends[0][0] if ends else 0.0
    t_req = max((period_ns * (1 - e[0] / e[5]) for e in ends), default=0.0)
    holds = [h[0] for h in holdl] + port_hold
    path = trace(ends[0]) if ends else []
    worst = {}
    for e in ends:
        worst.setdefault(e[1], round(e[0], 4))
    extra = {}
    if graph:
        extra["_graph"] = {"cell": ccell, "nin": nin, "nout": nout, "tin": tin, "tout": tout, "rel": rel, "rel_names": rel_names,
                           "dtab": dtab, "levels": levels, "A": A, "S": S, "load": load, "cpin": cpin, "tables": (X, Y, V),
                           "ends": ends, "tags": list(tags), "sinks": sinks, "drv": drv,
                           "end_req": {e: e[0] + float(A[e[6], RF.index(e[3]), e[2]]) + wget(e[2], e[4]) for e in ends}}
    return extra | {"period_ns": period_ns, "wns_ns": round(wns, 4), "tns_ns": round(sum(min(0.0, s) for s in worst.values()), 3),
            "endpoints": len(worst), "failing_endpoints": sum(s < 0 for s in worst.values()),
            "fmax_mhz": round(1000.0 / t_req, 1) if t_req > 0 else None,
            "whs_ns": round(min(holds), 4) if holds else None,
            "ths_ns": round(sum(min(0.0, s) for s in list(hold_by_ff.values()) + port_hold), 3),
            "hold_failing": sum(s < 0 for s in hold_by_ff.values()) + sum(s < 0 for s in port_hold),
            "clock": {"propagated": cx.propagated, "buffers": len(cx.clk_cells),
                      "latency_ns": [round(min(lats), 4), round(max(lats), 4)] if lats else None,
                      "skew_ns": round(max(lats) - min(lats), 4) if lats else None,
                      "domains": {c: round(v[0], 4) for c, v in clocks.items()}},
            "slew_violations": len(slow), "cap_violations": len(drv_cap),
            "critical_endpoint": ends[0][1] if ends else None, "critical_path": path,
            "critical_sink_idx": ends[0][4][0] if ends and ends[0][4][0] != "port" else None,
            "slack_by_endpoint": worst, "net_slack": net_slack, "hold_by_ff": hold_by_ff,
            "slow_nets": slow, "cap_nets": set(drv_cap)}
