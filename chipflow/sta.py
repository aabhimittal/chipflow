"""Back end: static timing analysis with Liberty NLDM tables.

Graph-based, single corner (typical). Rise and fall are propagated separately with unateness;
cell delay and output slew come from the cell_rise/cell_fall and *_transition tables at
(input slew, net load). Wire delay is an Elmore pi-model per sink: r*d*(c*d/2 + C_pin), with d
the driver-to-sink Manhattan distance and the net's wire cap set from its routed length when
available (else its placed HPWL).

Clock: ideal (zero latency) until the netlist has a clock tree; then it is propagated, so each
flip-flop launches and captures at its own clock latency and skew enters setup and hold.
Setup uses latest arrivals, hold earliest. Constraints (net.io, see chipflow.sdc) give ports
their delays, drivers and loads, define clocks, and add exceptions. Without constraints, inputs
arrive ideally at time 0, outputs are checked against the full period, and ports are left out of
hold.

Path tags: arrivals are kept per tag (launch clock, exceptions whose -from matched the start
point, exceptions whose -through nets the path crossed), so exceptions and clock relations are
exact per path, not per net. With one clock and no exceptions every net has one tag.
Launch/capture between clocks uses the most restrictive edge pair over their common period;
hold derives from that pair, as in SDC.
"""
from __future__ import annotations

import bisect
import math

from chipflow import pdk
from chipflow.sim import comb_order

IN_SLEW, CLK_SLEW, OUT_LOAD_FF = 0.02, 0.02, 2.0   # ns, ns, fF
MAX_SLEW = 0.198535                                 # Nangate45 default_max_transition, ns
RF = ("r", "f")
RISE = {"r": "rise", "f": "fall"}
NEVER = 1e9
CK, UNTIMED = 0, 1   # tag ids: clock-tree arrivals; propagated for slew only, never checked


def clock_network(net, lib, drv=None):
    """Cells and nets that distribute the clock: walk back from flip-flop clock pins through
    single-input buffers. Returns (cells, nets, root ports)."""
    drv = drv or net.drivers_sinks(lib)[0]
    cells, nets, roots = set(), set(), set()
    todo = [net.pins[c][lib[t].ff["clock"]] for c, (_, t) in enumerate(net.cells) if lib[t].ff]
    while todo:
        n = todo.pop()
        if n in nets:
            continue
        nets.add(n)
        d = drv.get(n)
        if d is None:
            continue
        if d[0] == "port":
            roots.add(d[1])
        elif lib[net.cells[d[0]][1]].name.startswith(("BUF", "CLKBUF")):
            cells.add(d[0])
            todo.append(net.pins[d[0]]["A"])
    return cells, nets, roots


def relation(launch, capture) -> tuple[float, float]:
    """(setup, hold) relation in ns between two clocks given as (period, rise offset)."""
    ps = lambda v: round(v * 1000)
    pl, ol, pc, oc = ps(launch[0]), ps(launch[1]), ps(capture[0]), ps(capture[1])
    n = pc // math.gcd(pl, pc)          # launch edges in the common period
    if n > 1000:
        raise ValueError(f"clock periods {launch[0]} and {capture[0]} ns have no short common period")
    best = None
    for i in range(n):
        le = ol + i * pl
        ce = oc + ((le - oc) // pc + 1) * pc   # first capture edge strictly after the launch edge
        if best is None or ce - le < best[0]:
            best = (ce - le, le, ce)
    s, le, ce = best
    return s / 1000, max(ce - pc - le, ce - (le + pl)) / 1000


_TEMPLATES: dict = {}


def _template(cell):
    """Per library cell, once: [(out pin, [(related pin, {t_out: (delay tab, slew tab, [t_in])})])]."""
    if id(cell) not in _TEMPLATES:
        outs = []
        for p in cell.outputs():
            arcs = []
            for arc in p.arcs:
                if not arc.kind.startswith("combinational"):
                    continue
                edges = {}
                for t_out in RF:
                    if f"cell_{RISE[t_out]}" in arc.tables:
                        ins = {"positive_unate": [t_out], "negative_unate": ["f" if t_out == "r" else "r"]}.get(arc.sense, list(RF))
                        edges[t_out] = (arc.tables[f"cell_{RISE[t_out]}"], arc.tables[f"{RISE[t_out]}_transition"],
                                        [(t_in, 2 * (t_in == "f") + (t_out == "f")) for t_in in ins])
                arcs.append((arc.related, edges))
            outs.append((p.name, arcs))
        _TEMPLATES[id(cell)] = (cell, outs)   # holding the cell keeps its id from being reused
    return _TEMPLATES[id(cell)][1]


def _arcs(net, lib, cells):
    """Per cell: [(out net, [(slot, in net, related pin, edges)])], slot unique per arc."""
    out, slot = {}, 0
    for c in cells:
        pins = net.pins[c]
        outs = []
        for pin, arcs in _template(lib[net.cells[c][1]]):
            if pin not in pins:
                continue
            cell_arcs = []
            for related, edges in arcs:
                if related in pins:
                    cell_arcs.append((slot, pins[related], related, edges))
                    slot += 1
            outs.append((pins[pin], cell_arcs))
        out[c] = outs
    return out


def analyze(net, lib, period_ns: float, pl=None, macros=None, net_len=None, hold=True) -> dict:
    """hold=False skips the earliest-arrival pass (inner loops that only need setup)."""
    drv, sinks = net.drivers_sinks(lib)
    r_um, c_um = pdk.WIRE_R_OHM_UM, pdk.WIRE_C_FF_UM

    def xy(p):
        if pl is None:
            return 0.0, 0.0
        return pl.io[(p[1], p[2])] if p[0] == "port" else pl.pin_xy(p[0], p[1], macros, net.cells[p[0]][1])

    io = net.io
    out_load = (lambda name: io["output_load"].get(name, 0.0)) if io else (lambda name: OUT_LOAD_FF)
    unc_setup, unc_hold = io["uncertainty"] if io else (0.0, 0.0)
    load, wire = {}, {}
    for n in range(net.n_nets):
        pins = sinks[n]
        cpin = sum(out_load(p[1]) if p[0] == "port" else lib[net.cells[p[0]][1]].pins[p[1]].cap for p in pins)
        if pl is not None and n in drv and pins:
            pts = [xy(drv[n])] + [xy(p) for p in pins]
            length = net_len.get(n) if net_len and n in net_len else (
                max(a for a, _ in pts) - min(a for a, _ in pts) + max(b for _, b in pts) - min(b for _, b in pts))
            dx0, dy0 = pts[0]
            for p, (sx, sy) in zip(pins, pts[1:]):
                d = abs(sx - dx0) + abs(sy - dy0)
                cp = out_load(p[1]) if p[0] == "port" else lib[net.cells[p[0]][1]].pins[p[1]].cap
                wire[(n, p)] = r_um * d * (c_um * d / 2 + cp) * 1e-6  # ohm*fF = 1e-6 ns
        else:
            length = 0.0
        load[n] = cpin + c_um * length

    clk_cells, clk_nets, clk_roots = clock_network(net, lib, drv)
    arc_of = {c: next(a for p in lib[c].outputs() for a in p.arcs if a.kind.startswith("combinational"))
              for c in set(io["driver"].values())} if io else {}
    propagated = bool(clk_cells)
    order = comb_order(net, lib)
    arcs = _arcs(net, lib, order)
    ffs = [c for c, (_, t) in enumerate(net.cells) if lib[t].ff]

    # clocks: name -> (period, rise), scaled so the first clock has period_ns (--clock-ns scales them all)
    if io and io.get("clocks"):
        primary = next(iter(io["clocks"]))
        k = period_ns / io["clocks"][primary]["period"]
        clocks = {c: (v["period"] * k, v["rise"] * k) for c, v in io["clocks"].items()}
        clock_of_port = {p: c for c, v in io["clocks"].items() for p in v["ports"]}
    else:
        primary, clocks, clock_of_port = "clk", {"clk": (period_ns, 0.0)}, {p: "clk" for p in clk_roots}
    rels = {}

    def rel(a, b):
        if (a, b) not in rels:
            rels[a, b] = relation(clocks[a], clocks[b])
        return rels[a, b]

    def root(n):
        while n in drv:
            d = drv[n]
            if d[0] == "port":
                return d[1]
            if d[0] not in clk_cells:
                return None
            n = net.pins[d[0]]["A"]
        return None

    ff_clock = {c: clock_of_port.get(root(net.pins[c][lib[net.cells[c][1]].ff["clock"]])) for c in ffs}
    groups = io.get("clock_groups", []) if io else []
    exs = io.get("exceptions", []) if io else []
    thr_of = {}
    for i, ex in enumerate(exs):
        for n in ex.get("through") or ():
            thr_of.setdefault(n, set()).add(i)
    thr_of = {n: frozenset(v) for n, v in thr_of.items()}
    from_ex = [i for i, ex in enumerate(exs) if ex.get("from") is not None]

    # tags are (launch clock, -from exception ids, -through exception ids), interned as small ints
    tags = [("~clock", frozenset(), frozenset()), (None, frozenset(), frozenset())]
    tag_id = {g: i for i, g in enumerate(tags)}

    def intern(g):
        if g not in tag_id:
            tag_id[g] = len(tags)
            tags.append(g)
            assert len(tags) < 4096, "too many path tags"   # packed into the delay-cache key below
        return tag_id[g]

    def start_tag(clk, key, n):
        if clk is None:
            return UNTIMED
        frm = frozenset(i for i in from_ex if key in exs[i]["from"] or ("clock", clk) in exs[i]["from"])
        return intern((clk, frm, thr_of.get(n, frozenset())))

    def bump(gid, n):
        b, g = thr_of.get(n), tags[gid]
        return gid if b is None or g[0] is None or b <= g[2] else intern((g[0], g[1], g[2] | b))

    def check(key, cap, gid):
        """None if this (start tag, endpoint) pair is not timed, else (setup N, hold M)."""
        g = tags[gid]
        if g[0] is None or gid == CK or cap is None:
            return None
        if any(any(g[0] in a for a in grp) and any(cap in b for b in grp) and
               not any(g[0] in a and cap in a for a in grp) for grp in groups):
            return None   # asynchronous clock groups
        N, M = 1, 0
        for i, ex in enumerate(exs):
            if ex.get("from") is not None and i not in g[1]:
                continue
            if ex.get("through") is not None and i not in g[2]:
                continue
            if ex.get("to") is not None and key not in ex["to"] and ("clock", cap) not in ex["to"]:
                continue
            if ex["kind"] == "false":
                return None
            N = ex["setup"] if ex.get("setup") is not None else N   # shortcut: later commands win, not SDC's specificity order
            M = ex["hold"] if ex.get("hold") is not None else M
        return N, M

    dcache = {}

    def forward(early: bool):
        at = {}
        for name, (d, nets) in net.ports.items():
            if d != "input":
                continue
            for n in nets:
                if name in clk_roots:
                    at[n] = {CK: {t: (0.0, IN_SLEW, None) for t in RF}}
                elif not io:  # no constraints: ideal driver at time 0, left out of hold
                    at[n] = {UNTIMED if early else start_tag(primary, ("port", name), n): {t: (0.0, IN_SLEW, None) for t in RF}}
                elif name not in io["input_delay"]:  # unconstrained: slew only
                    at[n] = {UNTIMED: {t: (0.0, IN_SLEW, None) for t in RF}}
                else:  # input delay, then the driving cell into this net's load
                    dl, arc = io["input_delay"][name][1 if early else 0], arc_of.get(io["driver"].get(name))
                    g = start_tag(io.get("input_clock", {}).get(name, primary), ("port", name), n)
                    at[n] = {g: {t: (dl + (arc.tables[f"cell_{RISE[t]}"](IN_SLEW, load[n]) if arc else 0.0),
                                     arc.tables[f"{RISE[t]}_transition"](IN_SLEW, load[n]) if arc else IN_SLEW, None) for t in RF}}
        for n in net.const:
            at[n] = {UNTIMED: {t: (0.0, IN_SLEW, None) for t in RF}}

        def comb(c):
            for n, cell_arcs in arcs[c]:
                out = {}
                ld = load[n]
                for slot, n_in, related, edges in cell_arcs:
                    din = at.get(n_in)
                    if not din:
                        continue
                    w = wire.get((n_in, (c, related)), 0.0)
                    for g_in, d_in in din.items():
                        g = bump(g_in, n) if thr_of else g_in
                        cur = out.get(g)
                        if cur is None:
                            cur = out[g] = {}
                        key = (slot << 14) | (g_in << 2)
                        for t_out, (dtab, stab, ins) in edges.items():
                            for t_in, code in ins:
                                if t_in not in d_in:
                                    continue
                                a_in, s_in, _ = d_in[t_in]
                                dly = dtab(s_in, ld)
                                if not early:
                                    dcache[key | code] = dly
                                a = a_in + w + dly
                                if t_out not in cur or (a < cur[t_out][0] if early else a > cur[t_out][0]):
                                    cur[t_out] = (a, stab(s_in, ld), (n_in, g_in, t_in, c, related, dly))
                at[n] = out

        for c in order:                     # the clock tree first: flip-flops launch off it
            if c in clk_cells:
                comb(c)
        lat = {}
        for c in ffs:
            cell = lib[net.cells[c][1]]
            ck = cell.ff["clock"]
            n_ck = net.pins[c][ck]
            ckd = at.get(n_ck, {}).get(CK)
            lat[c] = (ckd["r"][0] + wire.get((n_ck, (c, ck)), 0.0), ckd["r"][1]) if propagated and ckd else (0.0, CLK_SLEW)
            L, s_ck = lat[c]
            for p in cell.outputs():
                n = net.pins[c].get(p.name)
                if n is None:
                    continue
                arc = next(a for a in p.arcs if a.kind == "rising_edge")
                at[n] = {start_tag(ff_clock[c], c, n): {t: (L + arc.tables[f"cell_{RISE[t]}"](s_ck, load[n]),
                                                              arc.tables[f"{RISE[t]}_transition"](s_ck, load[n]),
                                                              ("launch", c, p.name)) for t in RF}}
        for c in order:
            if c not in clk_cells:
                comb(c)
        return at, lat

    at, lat = forward(early=False)

    def sink(at, n, p, g, t):
        a, s, _ = at[n][g][t]
        return a + wire.get((n, p), 0.0), s

    # setup endpoints: (slack, name, net, edge, sink pin, relation incl. multicycle, tag)
    ends = []
    for c in ffs:
        inst, typ = net.cells[c]
        cell = lib[typ]
        dpin = cell.ff["next"]
        arc = next((a for a in cell.pins[dpin].arcs if a.kind == "setup_rising"), None)
        n = net.pins[c][dpin]
        L, s_ck = lat[c]
        cap = ff_clock[c]
        for g, d in at.get(n, {}).items():
            chk = check(c, cap, g)
            if chk is None:
                continue
            R = rel(tags[g][0], cap)[0] + (chk[0] - 1) * clocks[cap][0]
            for t in d:
                a, s = sink(at, n, (c, dpin), g, t)
                su = arc.tables[f"{RISE[t]}_constraint"](s, s_ck) if arc else 0.0
                ends.append((R + L - su - a - unc_setup, f"{inst}/{dpin}", n, t, (c, dpin), R, g))
    for name, (d, nets) in net.ports.items():
        if d != "output" or (io and name not in io["output_delay"]):
            continue
        cap = io.get("output_clock", {}).get(name, primary) if io else primary
        od = io["output_delay"][name][0] if io else 0.0
        for b, n in enumerate(nets):
            for g, dd in at.get(n, {}).items():
                chk = check(("port", name), cap, g)
                if chk is None:
                    continue
                R = rel(tags[g][0], cap)[0] + (chk[0] - 1) * clocks[cap][0]
                for t in dd:
                    a, _ = sink(at, n, ("port", name, b), g, t)
                    ends.append((R - od - a - unc_setup, f"{name}[{b}]", n, t, ("port", name, b), R, g))
    ends.sort(key=lambda e: e[0])

    # backward pass: required times per tag, reusing the forward pass's delays -> slack on every net
    req = {}
    for sl, _, n, t, _, _, g in ends:
        r = req.setdefault(n, {}).setdefault(g, {})
        r[t] = min(r.get(t, math.inf), sl + at[n][g][t][0])
    for c in reversed(order):
        for n, cell_arcs in arcs[c]:
            rn = req.get(n)
            if not rn:
                continue
            for slot, n_in, related, edges in cell_arcs:
                w = wire.get((n_in, (c, related)), 0.0)
                for g_in, d_in in at.get(n_in, {}).items():
                    r_out = rn.get(bump(g_in, n) if thr_of else g_in)
                    if not r_out:
                        continue
                    r_in = req.setdefault(n_in, {}).setdefault(g_in, {})
                    key = (slot << 14) | (g_in << 2)
                    for t_out, (_, _, ins) in edges.items():
                        if t_out not in r_out:
                            continue
                        for t_in, code in ins:
                            if t_in in d_in:
                                v = r_out[t_out] - dcache[key | code] - w
                                if v < r_in.get(t_in, math.inf):
                                    r_in[t_in] = v
    net_slack = {n: min(r - at[n][g][t][0] for g, rg in rn.items() for t, r in rg.items())
                 for n, rn in req.items() if any(rn.values())}

    # hold: earliest data arrival vs. the capturing flip-flop's clock plus its hold time
    at_e = forward(early=True)[0] if hold else None
    hold = []
    port_hold = []  # outputs: the outside world holds for -min output delay after the edge
    if at_e is not None:
        for c in ffs:
            inst, typ = net.cells[c]
            cell = lib[typ]
            dpin = cell.ff["next"]
            arc = next((a for a in cell.pins[dpin].arcs if a.kind == "hold_rising"), None)
            n = net.pins[c][dpin]
            L, s_ck = lat[c]
            cap = ff_clock[c]
            for g, d in at_e.get(n, {}).items():
                chk = check(c, cap, g)
                if chk is None:
                    continue
                H = rel(tags[g][0], cap)[1] + (chk[0] - 1 - chk[1]) * clocks[cap][0]
                for t in d:
                    a, s = sink(at_e, n, (c, dpin), g, t)
                    h = arc.tables[f"{RISE[t]}_constraint"](s, s_ck) if arc else 0.0
                    hold.append((a - L - h - H - unc_hold, f"{inst}/{dpin}", c))
        for name, (d, nets) in (net.ports.items() if io else []):
            if d != "output" or name not in io["output_delay"]:
                continue
            cap = io.get("output_clock", {}).get(name, primary)
            for b, n in enumerate(nets):
                for g, dd in at_e.get(n, {}).items():
                    chk = check(("port", name), cap, g)
                    if chk is None:
                        continue
                    H = rel(tags[g][0], cap)[1] + (chk[0] - 1 - chk[1]) * clocks[cap][0]
                    a = min(sink(at_e, n, ("port", name, b), g, t)[0] for t in dd)
                    port_hold.append(a + io["output_delay"][name][1] - H - unc_hold)
    hold.sort()
    hold_by_ff = {}
    for sl, _, c in hold:
        hold_by_ff.setdefault(c, sl)

    drv_cap = {}
    for n, d in drv.items():
        if d[0] != "port":
            mc = lib[net.cells[d[0]][1]].pins[d[1]].max_cap
        elif io and d[1] not in clk_roots and d[1] in io["driver"]:
            mc = next(p.max_cap for p in lib[io["driver"][d[1]]].outputs())
        else:
            continue
        if load[n] > mc:
            drv_cap[n] = load[n]
    slow = {n for n, v in at.items() if n not in net.const and any(x[1] > MAX_SLEW for d in v.values() for x in d.values())}
    lats = [lat[c][0] for c in ffs]

    wns = ends[0][0] if ends else 0.0
    # scale every clock by s: an endpoint's slack is s*R + (slack - R), zero at s = 1 - slack/R
    t_req = max((period_ns * (1 - e[0] / e[5]) for e in ends), default=0.0)
    holds = [h[0] for h in hold] + port_hold
    path = _trace(net, at, wire, ends[0]) if ends else []
    worst_per_endpoint = {}
    for e in ends:  # sorted worst first: keep each endpoint's worse edge (rise or fall) and tag
        worst_per_endpoint.setdefault(e[1], round(e[0], 4))
    return {"period_ns": period_ns, "wns_ns": round(wns, 4), "tns_ns": round(sum(min(0.0, s) for s in worst_per_endpoint.values()), 3),
            "endpoints": len(worst_per_endpoint), "failing_endpoints": sum(s < 0 for s in worst_per_endpoint.values()),
            "fmax_mhz": round(1000.0 / t_req, 1) if t_req > 0 else None,
            "whs_ns": round(min(holds), 4) if holds else None,
            "ths_ns": round(sum(min(0.0, s) for s in list(hold_by_ff.values()) + port_hold), 3),
            "hold_failing": sum(s < 0 for s in hold_by_ff.values()) + sum(s < 0 for s in port_hold),
            "clock": {"propagated": propagated, "buffers": len(clk_cells),
                      "latency_ns": [round(min(lats), 4), round(max(lats), 4)] if lats else None,
                      "skew_ns": round(max(lats) - min(lats), 4) if lats else None,
                      "domains": {c: round(v[0], 4) for c, v in clocks.items()}},
            "slew_violations": len(slow), "cap_violations": len(drv_cap),
            "critical_endpoint": ends[0][1] if ends else None, "critical_path": path,
            "critical_sink_idx": ends[0][4][0] if ends and ends[0][4][0] != "port" else None,
            "slack_by_endpoint": worst_per_endpoint, "net_slack": net_slack, "hold_by_ff": hold_by_ff,
            "slow_nets": slow, "cap_nets": set(drv_cap)}


def _trace(net, at, wire, end):
    """Walk predecessor pointers back from the worst endpoint."""
    _, _, n, t, sink, _, g = end
    stages = []
    while True:
        a, _, pred = at[n][g][t]
        wd = wire.get((n, sink), 0.0)
        if pred is None:
            stages.append({"point": "input", "arrival_ns": round(a, 4)})
            break
        if pred[0] == "launch":
            c, pin = pred[1], pred[2]
            stages.append({"idx": c, "point": f"{net.cells[c][0]}/{pin}", "cell": net.cells[c][1], "edge": t,
                           "incr_ns": round(a, 4), "wire_ns": round(wd, 4), "arrival_ns": round(a, 4)})
            break
        n_in, g_in, t_in, c, rel_pin, dly = pred
        stages.append({"idx": c, "point": f"{net.cells[c][0]}", "cell": net.cells[c][1], "edge": t,
                       "incr_ns": round(dly, 4), "wire_ns": round(wd, 4), "arrival_ns": round(a, 4)})
        n, g, t, sink = n_in, g_in, t_in, (c, rel_pin)
    return stages[::-1]
