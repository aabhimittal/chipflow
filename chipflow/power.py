"""Power: switching power from simulated toggle rates on routed net capacitance, plus Liberty leakage.

P_switch = sum_n rate_n * 0.5 * C_n * V^2 * f, with rate_n = transitions per cycle measured by the
gate-level simulator (the clock net: 2 per cycle). Cell internal power is not modeled yet
(shortcut: Liberty internal_power tables are not parsed, so total power is an underestimate).
"""
from __future__ import annotations

from chipflow import pdk


def analyze(net, lib, voltage, period_ns, toggles, cycles, lanes, net_len=None) -> dict:
    _, sinks = net.drivers_sinks(lib)
    clock_nets = {net.pins[c][lib[t].ff["clock"]] for c, (_, t) in enumerate(net.cells) if lib[t].ff}
    f_hz = 1e9 / period_ns
    sw = clk = 0.0
    for n in range(net.n_nets):
        cap = sum(2.0 if p[0] == "port" else lib[net.cells[p[0]][1]].pins[p[1]].cap for p in sinks[n])
        cap += pdk.WIRE_C_FF_UM * (net_len or {}).get(n, 0.0)
        rate = 2.0 if n in clock_nets else toggles[n] / max(1, cycles * lanes)
        p = rate * 0.5 * cap * 1e-15 * voltage ** 2 * f_hz
        if n in clock_nets:
            clk += p
        else:
            sw += p
    leak = sum(lib[t].leakage_nw for _, t in net.cells) * 1e-9
    mw = lambda w: round(w * 1e3, 4)
    return {"switching_mw": mw(sw), "clock_mw": mw(clk), "leakage_mw": mw(leak), "total_mw": mw(sw + clk + leak),
            "note": "excludes cell internal power", "activity_cycles": cycles, "activity_lanes": lanes}
