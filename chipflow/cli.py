"""chipflow: Verilog to placed-and-routed layout on Nangate45.

  python -m chipflow fetch-pdk
  python -m chipflow flow --design systolic --param N=4 --clock-ns 2.0 --out runs/sa4
  python -m chipflow flow --verilog my.v --top my_top --clock-ns 1.5 --out runs/my
  python -m chipflow flow --design systolic --sdc designs/systolic_array.sdc --out runs/sa4_sdc
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chipflow import buffer, designs, layout, pdk, place, power, route, sdc, sim, size, sta, synth


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _summary(t):
    return {k: v for k, v in t.items() if k not in ("critical_path", "slack_by_endpoint", "net_slack", "hold_by_ff", "slow_nets", "cap_nets")}


def flow(a):
    """Run every stage; returns (report, internals) so tools like demo/build_data.py can reuse them."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = print
    t0 = time.time()
    pdk.fetch()
    L = pdk.load_liberty()
    lib, lef = L["cells"], pdk.load_lef()
    if a.design:
        spec = designs.DESIGNS[a.design]
        sources, top = spec["sources"], spec["top"]
        params = dict(spec["params"])
    else:
        sources, top, params, spec = a.verilog, a.top, {}, None
    for kv in a.param or []:
        k, v = kv.split("=")
        params[k] = int(v)
    rep = {"design": top, "params": params}

    log(f"[1/8] synthesis: {top} {params}")
    net, rep["synth"] = synth.synthesize(sources, top, params, out)
    if a.sdc:   # constraints file; its create_clock sets the period unless --clock-ns is given
        net.io, period = sdc.read(a.sdc, net, lib)
        a.clock_ns = a.clock_ns or period or 2.0
        rep["io"] = {"sdc": Path(a.sdc).name} | sdc.summary(net.io)
    else:
        a.clock_ns = a.clock_ns or 2.0
        if a.io:  # default budget: the world outside the block uses io_budget of the cycle on each side
            d = a.io_budget * a.clock_ns
            net.io = sdc.budget(net, d, a.input_driver, a.output_load)
            rep["io"] = {"input_delay": d, "output_delay": d, "driver": a.input_driver, "output_load": a.output_load}
    rep["clock_ns"] = a.clock_ns
    log(f"      {rep['synth']['cells']} cells, {rep['synth']['area_um2']:.0f} um2, {rep['synth']['flip_flops']} flip-flops, "
        f"{len(rep['synth']['lint_warnings'])} lint warnings")

    def simulate(tag):
        s = sim.Simulator(net, lib)
        rng = np.random.default_rng(a.seed)
        if spec and spec.get("tb"):
            res = spec["tb"](s, params, rng)
            log(f"      {tag}: {res['mismatches']} mismatches in {res['checked']} checked outputs vs golden model")
        else:
            for _ in range(32):
                s.step({p: rng.integers(0, 2 ** 63, size=len(ns), dtype=np.uint64)
                        for p, (d, ns) in net.ports.items() if d == "input"})
            res = {"checked": 0, "mismatches": None, "cycles": s.cycles, "note": "random stimulus, activity only"}
            log(f"      {tag}: no golden model, random stimulus for switching activity only")
        return res, s

    log("[2/8] gate-level simulation")
    rep["verify"], s = simulate("synthesized netlist")

    log("[3/8] floorplan + global placement + legalization")
    pl = place.place(net, lib, lef, util=a.util, seed=a.seed, log=log)
    rep["place"] = {k: v for k, v in pl.stats.items() if k != "gp_hist"}
    log(f"      core {pl.stats['core_um']} um, HPWL {pl.stats['hpwl_um']:.0f} um, legalization moved {pl.stats['legal_disp_avg_um']} um avg")
    t_pl = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"], hold=False)
    weights = None
    if a.timing_driven:  # weight near-critical nets, re-place from the current solution
        weights = place.timing_weights(t_pl, a.clock_ns)
        log(f"      fmax {t_pl['fmax_mhz']} MHz; timing-driven re-placement weighting {len(weights)} nets")
        pl = place.place(net, lib, lef, util=a.util, seed=a.seed, init=np.stack([pl.x + pl.w / 2, pl.y + pl.h / 2], 1), net_weight=weights)
        rep["place"] = {k: v for k, v in pl.stats.items() if k != "gp_hist"} | {"timing_driven_nets": len(weights)}

    log("[4/8] clock tree, design-rule repair, timing repair, hold")
    if a.cts:
        rep["cts"] = buffer.cts(net, lib, lef, pl)
        log(f"      clock tree: {rep['cts']['buffers']} {rep['cts']['cell']} buffers")
    if a.buffer:
        rep["repair"] = buffer.repair_design(net, lib, lef, pl, a.clock_ns)
        log(f"      design-rule repair: {rep['repair']['nets_buffered']} nets buffered ({rep['repair']['buffers']} buffers), "
            f"{rep['repair']['drivers_upsized']} drivers upsized; slew/cap violations "
            f"{rep['repair']['violations_before']['slew']}/{rep['repair']['violations_before']['cap']} -> "
            f"{rep['repair']['violations_after']['slew']}/{rep['repair']['violations_after']['cap']}")
    t_pl = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"], hold=False)
    if a.size and t_pl["wns_ns"] < 0:
        rep["sizing"] = size.upsize(net, lib, lef, pl, a.clock_ns, log=log)
        log(f"      gate sizing: {rep['sizing']['cells_upsized']} cells upsized, WNS {rep['sizing']['wns_before_ns']:+.3f} -> {rep['sizing']['wns_after_ns']:+.3f} ns")
    if a.cts:
        rep["hold_fix"] = buffer.fix_hold(net, lib, lef, pl, a.clock_ns)
        log(f"      hold: worst slack {rep['hold_fix']['whs_after_ns']:+.3f} ns, {rep['hold_fix']['buffers']} delay buffers")
    t_pl = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"])
    rep["sta_placed"] = _summary(t_pl)
    log(f"      WNS {t_pl['wns_ns']:+.3f} ns, fmax {t_pl['fmax_mhz']} MHz, skew {t_pl['clock']['skew_ns']} ns")
    rep["verify_backend"], s = simulate("re-verified after clock tree, buffering and sizing")

    log("[5/8] global routing")
    rt = route.route(net, lib, pl, lef, log=log)
    rounds = 0
    while rt["overflow"] and rounds < a.routability_rounds:  # inflate congested cells, re-place, reroute
        rounds += 1
        f = route.inflation(pl, rt["grid_obj"])
        log(f"      overflow {rt['overflow']}: routability round {rounds}, inflating {int((f > 1).sum())} cells and re-placing")
        pl = place.place(net, lib, lef, util=a.util, seed=a.seed, init=np.stack([pl.x + pl.w / 2, pl.y + pl.h / 2], 1),
                         inflate=f, net_weight=weights)
        rt = route.route(net, lib, pl, lef)
    if rounds:
        rep["place"] = {k: v for k, v in pl.stats.items() if k != "gp_hist"} | {
            k: rep["place"][k] for k in ("timing_driven_nets",) if k in rep["place"]}
        rep["sta_placed"] = _summary(sta.analyze(net, lib, a.clock_ns, pl, lef["macros"]))
    rep["route"] = {k: v for k, v in rt.items() if k not in ("net_len", "grid_obj", "conns")} | {"routability_rounds": rounds}
    log(f"      wirelength {rt['wirelength_um']:.0f} um, overflow {rt['overflow']}, peak edge use {rt['max_util']:.2f}")

    log("[6/8] timing after routing")
    t_rt = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"], rt["net_len"])
    if a.size and t_rt["wns_ns"] < 0:  # post-route repair on routed lengths, then reroute the moved cells
        rep["sizing_post_route"] = size.upsize(net, lib, lef, pl, a.clock_ns, log=log, net_len=rt["net_len"])
        rt = route.route(net, lib, pl, lef)
        rep["route"] |= {k: v for k, v in rt.items() if k not in ("net_len", "grid_obj", "conns")}
        t_rt = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"], rt["net_len"])
        log(f"      post-route sizing: {rep['sizing_post_route']['cells_upsized']} cells, rerouted overflow {rt['overflow']}")
    if a.cts and t_rt["hold_failing"]:  # post-route hold repair on routed lengths, then reroute
        rep["hold_fix_post_route"] = buffer.fix_hold(net, lib, lef, pl, a.clock_ns, net_len=rt["net_len"])
        rt = route.route(net, lib, pl, lef)
        rep["route"] |= {k: v for k, v in rt.items() if k not in ("net_len", "grid_obj", "conns")}
        t_rt = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"], rt["net_len"])
        log(f"      post-route hold fix: {rep['hold_fix_post_route']['buffers']} delay buffers, rerouted overflow {rt['overflow']}")
    rep["sta_routed"] = _summary(t_rt)
    rep["critical_path"] = [{k: v for k, v in st.items() if k != "idx"} for st in t_rt["critical_path"]]
    log(f"      setup WNS {t_rt['wns_ns']:+.3f} ns, TNS {t_rt['tns_ns']} ns, fmax {t_rt['fmax_mhz']} MHz, {len(t_rt['critical_path'])} stages; "
        f"hold WHS {t_rt['whs_ns']:+.3f} ns ({t_rt['hold_failing']} failing); clock skew {t_rt['clock']['skew_ns']} ns")

    log("[7/8] power")
    rep["power"] = power.analyze(net, lib, L["voltage"], a.clock_ns, s.toggles, s.cycles, 64, rt["net_len"])
    log(f"      {rep['power']['total_mw']} mW (clock {rep['power']['clock_mw']} mW)")

    log("[8/8] outputs")
    layout.write_def(out / "placed.def", net, pl, lef)
    layout.write_svg(out / "layout.svg", net, pl, lib, [st["idx"] for st in t_rt["critical_path"] if "idx" in st])
    layout.write_congestion_svg(out / "congestion.svg", rt["grid_obj"])
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(_clean(rep), indent=1))
    log(f"      {out}/report.json, layout.svg, congestion.svg, placed.def, netlist.v  ({rep['seconds']} s)")
    return rep, {"net": net, "lib": lib, "lef": lef, "pl": pl, "route": rt, "sta": t_rt, "params": params,
                 "sources": sources}


def main(argv=None):
    p = argparse.ArgumentParser("chipflow")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-pdk")
    f = sub.add_parser("flow")
    src = f.add_mutually_exclusive_group(required=True)
    src.add_argument("--design", choices=sorted(designs.DESIGNS))
    src.add_argument("--verilog", nargs="+")
    f.add_argument("--top")
    f.add_argument("--param", action="append", help="NAME=INT parameter override")
    f.add_argument("--clock-ns", type=float, default=None, help="clock period (default: the SDC's create_clock, else 2.0)")
    f.add_argument("--sdc", help="timing constraints file (replaces --io-budget/--input-driver/--output-load)")
    f.add_argument("--util", type=float, default=0.6)
    f.add_argument("--routability-rounds", type=int, default=2, help="re-place congested regions up to N times")
    f.add_argument("--timing-driven", action=argparse.BooleanOptionalAction, default=True, help="re-place with critical nets weighted")
    f.add_argument("--size", action=argparse.BooleanOptionalAction, default=True, help="upsize cells when timing fails")
    f.add_argument("--cts", action=argparse.BooleanOptionalAction, default=True, help="build a clock tree, check and fix hold")
    f.add_argument("--io", action=argparse.BooleanOptionalAction, default=True, help="time ports with I/O constraints")
    f.add_argument("--io-budget", type=float, default=0.3, help="input and output delay as a fraction of the clock period")
    f.add_argument("--input-driver", default="BUF_X4", help="library cell assumed to drive each input port")
    f.add_argument("--output-load", type=float, default=4.0, help="load on each output port, fF")
    f.add_argument("--buffer", action=argparse.BooleanOptionalAction, default=True, help="buffer high-fanout nets, fix slew/cap limits")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", default="runs/latest")
    a = p.parse_args(argv)
    if a.cmd == "fetch-pdk":
        print(pdk.fetch())
    else:
        if a.verilog and not a.top:
            p.error("--verilog needs --top")
        flow(a)  # returns (report, internals); the CLI only needs the files it writes


if __name__ == "__main__":
    main()
