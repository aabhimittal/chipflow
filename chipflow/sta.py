"""Back end: static timing analysis with Liberty NLDM tables.

Graph-based, single corner (typical). Rise and fall are propagated separately with unateness;
cell delay and output slew come from the cell_rise/cell_fall and *_transition tables at
(input slew, net load). Wire delay is an Elmore pi-model per sink: r*d*(c*d/2 + C_pin), with d
the driver-to-sink Manhattan distance and the net's wire cap set from its routed length when
available (else its placed HPWL).

Clock: ideal (zero latency) until the netlist has a clock tree; then it is propagated, so each
flip-flop launches and captures at its own clock latency and skew enters setup and hold.
Setup uses latest arrivals, hold earliest. Constraints (net.io, see chipflow.sdc) give ports
their delays, drivers and loads and add exceptions: false paths drop start or end points,
multicycle paths move an endpoint's capture edge. Without constraints, inputs arrive ideally at
time 0, outputs are checked against the full period, and ports are left out of hold.
"""
from __future__ import annotations

from chipflow import pdk
from chipflow.sim import comb_order

IN_SLEW, CLK_SLEW, OUT_LOAD_FF = 0.02, 0.02, 2.0   # ns, ns, fF
MAX_SLEW = 0.198535                                 # Nangate45 default_max_transition, ns
RF = ("r", "f")
RISE = {"r": "rise", "f": "fall"}
NEVER = 1e9


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


def analyze(net, lib, period_ns: float, pl=None, macros=None, net_len=None, hold=True) -> dict:
    """hold=False skips the earliest-arrival pass (inner loops that only need setup)."""
    drv, sinks = net.drivers_sinks(lib)
    r_um, c_um = pdk.WIRE_R_OHM_UM, pdk.WIRE_C_FF_UM

    def xy(p):
        if pl is None:
            return 0.0, 0.0
        return pl.io[(p[1], p[2])] if p[0] == "port" else pl.pin_xy(p[0], p[1], macros, net.cells[p[0]][1])

    io = net.io
    T = period_ns
    out_load = (lambda name: io["output_load"].get(name, 0.0)) if io else (lambda name: OUT_LOAD_FF)
    mcp = (lambda k: io["multicycle"].get(k, (1, 0))) if io else (lambda k: (1, 0))
    false_from, false_to = (io["false_from"], io["false_to"]) if io else (set(), set())
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
    ffs = [c for c, (_, t) in enumerate(net.cells) if lib[t].ff]

    def forward(early: bool):
        at = {}
        better = (lambda a, b: a < b) if early else (lambda a, b: a > b)
        for name, (d, nets) in net.ports.items():
            if d != "input":
                continue
            for n in nets:
                if name in clk_roots:
                    at[n] = {t: (0.0, IN_SLEW, None) for t in RF}
                elif not io:  # no constraints: ideal driver at time 0, left out of hold
                    at[n] = {t: (NEVER if early else 0.0, IN_SLEW, None) for t in RF}
                elif name not in io["input_delay"] or ("port", name) in false_from:  # not timed
                    at[n] = {t: (NEVER if early else -NEVER, IN_SLEW, None) for t in RF}
                else:  # input delay, then the driving cell into this net's load
                    d, arc = io["input_delay"][name][1 if early else 0], arc_of.get(io["driver"].get(name))
                    at[n] = {t: (d + (arc.tables[f"cell_{RISE[t]}"](IN_SLEW, load[n]) if arc else 0.0),
                                 arc.tables[f"{RISE[t]}_transition"](IN_SLEW, load[n]) if arc else IN_SLEW, None) for t in RF}
        for n in net.const:
            at[n] = {t: (NEVER if early else 0.0, IN_SLEW, None) for t in RF}

        def sink_arrival(n, p, t):
            a, s, _ = at[n][t]
            return a + wire.get((n, p), 0.0), s

        def comb(c):
            inst, typ = net.cells[c]
            for p in lib[typ].outputs():
                n = net.pins[c].get(p.name)
                if n is None:
                    continue
                best = {}
                for arc in p.arcs:
                    if not arc.kind.startswith("combinational") or arc.related not in net.pins[c]:
                        continue
                    n_in = net.pins[c][arc.related]
                    for t_out in RF:
                        tab = "rise" if t_out == "r" else "fall"
                        if f"cell_{tab}" not in arc.tables:
                            continue
                        ins = {"positive_unate": [t_out], "negative_unate": ["f" if t_out == "r" else "r"]}.get(arc.sense, list(RF))
                        for t_in in ins:
                            a_in, s_in = sink_arrival(n_in, (c, arc.related), t_in)
                            dly = arc.tables[f"cell_{tab}"](s_in, load[n])
                            if t_out not in best or better(a_in + dly, best[t_out][0]):
                                best[t_out] = (a_in + dly, arc.tables[f"{tab}_transition"](s_in, load[n]),
                                               (n_in, t_in, c, arc.related, dly))
                at[n] = {t: best.get(t, (NEVER if early else 0.0, IN_SLEW, None)) for t in RF}

        for c in order:                     # the clock tree first: flip-flops launch off it
            if c in clk_cells:
                comb(c)
        lat = {}
        for c in ffs:
            cell = lib[net.cells[c][1]]
            ck = cell.ff["clock"]
            n_ck = net.pins[c][ck]
            lat[c] = sink_arrival(n_ck, (c, ck), "r") if propagated else (0.0, CLK_SLEW)
            for p in cell.outputs():
                n = net.pins[c].get(p.name)
                if n is None:
                    continue
                arc = next(a for a in p.arcs if a.kind == "rising_edge")
                L, s_ck = lat[c]
                L = (NEVER if early else -NEVER) if c in false_from else L   # false path from this register
                at[n] = {t: (L + arc.tables[f"cell_{'rise' if t == 'r' else 'fall'}"](s_ck, load[n]),
                             arc.tables[f"{'rise' if t == 'r' else 'fall'}_transition"](s_ck, load[n]),
                             ("launch", c, p.name)) for t in RF}
        for c in order:
            if c not in clk_cells:
                comb(c)
        return at, lat, sink_arrival

    at, lat, sink_arrival = forward(early=False)

    ends = []
    for c in ffs:
        inst, typ = net.cells[c]
        cell = lib[typ]
        dpin = cell.ff["next"]
        arc = next((a for a in cell.pins[dpin].arcs if a.kind == "setup_rising"), None)
        n = net.pins[c][dpin]
        L, s_ck = lat[c]
        N = mcp(c)[0]
        for t in RF:
            a, s = sink_arrival(n, (c, dpin), t)
            if c in false_to or a < -NEVER / 2:
                continue
            su = arc.tables[f"{'rise' if t == 'r' else 'fall'}_constraint"](s, s_ck) if arc else 0.0
            ends.append((N * T + L - su - a - unc_setup, f"{inst}/{dpin}", n, t, (c, dpin), N))
    for name, (d, nets) in net.ports.items():
        if d != "output" or ("port", name) in false_to or (io and name not in io["output_delay"]):
            continue
        N = mcp(("port", name))[0]
        for b, n in enumerate(nets):
            if n in at:
                for t in RF:
                    a, _ = sink_arrival(n, ("port", name, b), t)
                    if a > -NEVER / 2:
                        od = io["output_delay"][name][0] if io else 0.0
                        ends.append((N * T - od - a - unc_setup, f"{name}[{b}]", n, t, ("port", name, b), N))
    ends.sort(key=lambda e: e[0])

    # backward pass: required times with the same delays as the forward pass -> slack on every net
    INF = float("inf")
    req = {n: {t: INF for t in RF} for n in at}
    for sl, _, n, t, *_ in ends:
        req[n][t] = min(req[n][t], sl + at[n][t][0])
    for c in reversed(order):
        cell = lib[net.cells[c][1]]
        for p in cell.outputs():
            n = net.pins[c].get(p.name)
            if n is None:
                continue
            for arc in p.arcs:
                if not arc.kind.startswith("combinational") or arc.related not in net.pins[c]:
                    continue
                n_in = net.pins[c][arc.related]
                for t_out in RF:
                    tab = "rise" if t_out == "r" else "fall"
                    if f"cell_{tab}" not in arc.tables or req[n][t_out] == INF:
                        continue
                    ins = {"positive_unate": [t_out], "negative_unate": ["f" if t_out == "r" else "r"]}.get(arc.sense, list(RF))
                    for t_in in ins:
                        dly = arc.tables[f"cell_{tab}"](at[n_in][t_in][1], load[n])
                        r_in = req[n][t_out] - dly - wire.get((n_in, (c, arc.related)), 0.0)
                        if r_in < req[n_in][t_in]:
                            req[n_in][t_in] = r_in
    net_slack = {n: min(req[n][t] - at[n][t][0] for t in RF) for n in at if min(req[n].values()) < INF}

    # hold: earliest data arrival vs. the capturing flip-flop's clock plus its hold time
    at_e, _, sink_e = forward(early=True) if hold else ({}, None, None)
    hold = []
    for c in (ffs if sink_e else []):
        inst, typ = net.cells[c]
        cell = lib[typ]
        dpin = cell.ff["next"]
        arc = next((a for a in cell.pins[dpin].arcs if a.kind == "hold_rising"), None)
        n = net.pins[c][dpin]
        L, s_ck = lat[c]
        N, M = mcp(c)
        for t in RF:
            a, s = sink_e(n, (c, dpin), t)
            if a >= NEVER / 2 or c in false_to:
                continue
            h = arc.tables[f"{'rise' if t == 'r' else 'fall'}_constraint"](s, s_ck) if arc else 0.0
            hold.append((a - L - h - (N - 1 - M) * T - unc_hold, f"{inst}/{dpin}", c))
    hold.sort()
    port_hold = []  # outputs: the outside world holds for -min output delay after the edge
    for name, (d, nets) in (net.ports.items() if io and sink_e else []):
        if d != "output" or name not in io["output_delay"] or ("port", name) in false_to:
            continue
        N, M = mcp(("port", name))
        for b, n in enumerate(nets):
            a = min(sink_e(n, ("port", name, b), t)[0] for t in RF) if n in at_e else NEVER
            if a < NEVER / 2:
                port_hold.append(a + io["output_delay"][name][1] - (N - 1 - M) * T - unc_hold)
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
    slow = {n for n, v in at.items() if n not in net.const and max(v[t][1] for t in RF) > MAX_SLEW}
    lats = [lat[c][0] for c in ffs]

    wns = ends[0][0] if ends else 0.0
    t_req = max((T - e[0] / e[5] for e in ends), default=0.0)   # period at which each endpoint has zero slack
    holds = [h[0] for h in hold] + port_hold
    path = _trace(net, at, wire, ends[0]) if ends else []
    worst_per_endpoint = {}
    for e in ends:  # sorted worst first: keep each endpoint's worse edge (rise or fall)
        worst_per_endpoint.setdefault(e[1], round(e[0], 4))
    return {"period_ns": period_ns, "wns_ns": round(wns, 4), "tns_ns": round(sum(min(0.0, e[0]) for e in ends), 3),
            "endpoints": len(ends) // 2, "failing_endpoints": len({e[1] for e in ends if e[0] < 0}),
            "fmax_mhz": round(1000.0 / t_req, 1) if t_req > 0 else None,
            "whs_ns": round(min(holds), 4) if holds else None,
            "ths_ns": round(sum(min(0.0, s) for s in list(hold_by_ff.values()) + port_hold), 3),
            "hold_failing": sum(s < 0 for s in hold_by_ff.values()) + sum(s < 0 for s in port_hold),
            "clock": {"propagated": propagated, "buffers": len(clk_cells),
                      "latency_ns": [round(min(lats), 4), round(max(lats), 4)] if lats else None,
                      "skew_ns": round(max(lats) - min(lats), 4) if lats else None},
            "slew_violations": len(slow), "cap_violations": len(drv_cap),
            "critical_endpoint": ends[0][1] if ends else None, "critical_path": path,
            "critical_sink_idx": ends[0][4][0] if ends and ends[0][4][0] != "port" else None,
            "slack_by_endpoint": worst_per_endpoint, "net_slack": net_slack, "hold_by_ff": hold_by_ff,
            "slow_nets": slow, "cap_nets": set(drv_cap)}


def _trace(net, at, wire, end):
    """Walk predecessor pointers back from the worst endpoint."""
    _, _, n, t, sink, _ = end
    stages = []
    while True:
        a, _, pred = at[n][t]
        wd = wire.get((n, sink), 0.0)
        if pred is None:
            stages.append({"point": "input", "arrival_ns": round(a, 4)})
            break
        if pred[0] == "launch":
            c, pin = pred[1], pred[2]
            stages.append({"idx": c, "point": f"{net.cells[c][0]}/{pin}", "cell": net.cells[c][1], "edge": t,
                           "incr_ns": round(a, 4), "wire_ns": round(wd, 4), "arrival_ns": round(a, 4)})
            break
        n_in, t_in, c, rel, dly = pred
        stages.append({"idx": c, "point": f"{net.cells[c][0]}", "cell": net.cells[c][1], "edge": t,
                       "incr_ns": round(dly, 4), "wire_ns": round(wd, 4), "arrival_ns": round(a, 4)})
        n, t, sink = n_in, t_in, (c, rel)
    return stages[::-1]
