"""Back end, step 3: static timing analysis with Liberty NLDM tables.

Graph-based, single corner (typical), ideal clock (no CTS yet). Rise and fall are propagated
separately with unateness; cell delay and output slew come from the cell_rise/cell_fall and
*_transition tables at (input slew, net load). Wire delay is an Elmore pi-model per sink:
r*d*(c*d/2 + C_pin), with d the driver-to-sink Manhattan distance and the net's wire cap set
from its routed length when available (else its placed HPWL).
"""
from __future__ import annotations

from chipflow import pdk
from chipflow.sim import comb_order

IN_SLEW, CLK_SLEW, OUT_LOAD_FF = 0.02, 0.02, 2.0   # ns, ns, fF
RF = ("r", "f")


def analyze(net, lib, period_ns: float, pl=None, macros=None, net_len=None) -> dict:
    drv, sinks = net.drivers_sinks(lib)
    r_um, c_um = pdk.WIRE_R_OHM_UM, pdk.WIRE_C_FF_UM

    def xy(p):
        if pl is None:
            return 0.0, 0.0
        return pl.io[(p[1], p[2])] if p[0] == "port" else pl.pin_xy(p[0], p[1], macros, net.cells[p[0]][1])

    load, wire = {}, {}
    for n in range(net.n_nets):
        pins = sinks[n]
        cpin = sum(OUT_LOAD_FF if p[0] == "port" else lib[net.cells[p[0]][1]].pins[p[1]].cap for p in pins)
        if pl is not None and n in drv and pins:
            pts = [xy(drv[n])] + [xy(p) for p in pins]
            length = net_len.get(n) if net_len and n in net_len else (
                max(a for a, _ in pts) - min(a for a, _ in pts) + max(b for _, b in pts) - min(b for _, b in pts))
            dx0, dy0 = pts[0]
            for p, (sx, sy) in zip(pins, pts[1:]):
                d = abs(sx - dx0) + abs(sy - dy0)
                cp = OUT_LOAD_FF if p[0] == "port" else lib[net.cells[p[0]][1]].pins[p[1]].cap
                wire[(n, p)] = r_um * d * (c_um * d / 2 + cp) * 1e-6  # ohm*fF = 1e-6 ns
        else:
            length = 0.0
        load[n] = cpin + c_um * length

    at = {}   # net -> {"r": (arrival, slew, pred), "f": ...}
    for name, (d, nets) in net.ports.items():
        if d == "input":
            for n in nets:
                at[n] = {t: (0.0, IN_SLEW, None) for t in RF}
    for n in net.const:
        at[n] = {t: (0.0, IN_SLEW, None) for t in RF}
    for c, (inst, typ) in enumerate(net.cells):
        cell = lib[typ]
        if not cell.ff:
            continue
        for p in cell.outputs():
            n = net.pins[c].get(p.name)
            if n is None:
                continue
            arc = next(a for a in p.arcs if a.kind == "rising_edge")
            at[n] = {t: (arc.tables[f"cell_{'rise' if t == 'r' else 'fall'}"](CLK_SLEW, load[n]),
                         arc.tables[f"{'rise' if t == 'r' else 'fall'}_transition"](CLK_SLEW, load[n]),
                         ("launch", c, p.name)) for t in RF}

    def sink_arrival(n, p, t):
        a, s, _ = at[n][t]
        return a + wire.get((n, p), 0.0), s

    for c in comb_order(net, lib):
        inst, typ = net.cells[c]
        cell = lib[typ]
        for p in cell.outputs():
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
                        if t_out not in best or a_in + dly > best[t_out][0]:
                            best[t_out] = (a_in + dly, arc.tables[f"{tab}_transition"](s_in, load[n]),
                                           (n_in, t_in, c, arc.related, dly))
            at[n] = {t: best.get(t, (0.0, IN_SLEW, None)) for t in RF}

    ends = []
    for c, (inst, typ) in enumerate(net.cells):
        cell = lib[typ]
        if not cell.ff:
            continue
        dpin = cell.ff["next"]
        arc = next((a for a in cell.pins[dpin].arcs if a.kind == "setup_rising"), None)
        n = net.pins[c][dpin]
        for t in RF:
            a, s = sink_arrival(n, (c, dpin), t)
            su = arc.tables[f"{'rise' if t == 'r' else 'fall'}_constraint"](s, CLK_SLEW) if arc else 0.0
            ends.append((period_ns - su - a, f"{inst}/{dpin}", n, t, (c, dpin)))
    for name, (d, nets) in net.ports.items():
        if d == "output":
            for b, n in enumerate(nets):
                if n in at:
                    for t in RF:
                        a, _ = sink_arrival(n, ("port", name, b), t)
                        ends.append((period_ns - a, f"{name}[{b}]", n, t, ("port", name, b)))
    ends.sort(key=lambda e: e[0])
    wns = ends[0][0] if ends else 0.0
    path = _trace(net, at, wire, ends[0]) if ends else []
    worst_per_endpoint = {}
    for e in ends:  # sorted worst first: keep each endpoint's worse edge (rise or fall)
        worst_per_endpoint.setdefault(e[1], round(e[0], 4))
    return {"period_ns": period_ns, "wns_ns": round(wns, 4), "tns_ns": round(sum(min(0.0, e[0]) for e in ends), 3),
            "endpoints": len(ends) // 2, "failing_endpoints": len({e[1] for e in ends if e[0] < 0}),
            "fmax_mhz": round(1000.0 / (period_ns - wns), 1) if period_ns - wns > 0 else None,
            "critical_endpoint": ends[0][1] if ends else None, "critical_path": path,
            "critical_sink_idx": ends[0][4][0] if ends and ends[0][4][0] != "port" else None,
            "slack_by_endpoint": worst_per_endpoint}


def _trace(net, at, wire, end):
    """Walk predecessor pointers back from the worst endpoint."""
    _, _, n, t, sink = end
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
