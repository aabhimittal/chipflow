"""chipflow: Verilog to placed-and-routed layout on Nangate45.

  python -m chipflow fetch-pdk
  python -m chipflow flow --design systolic --param N=4 --clock-ns 2.0 --out runs/sa4
  python -m chipflow flow --verilog my.v --top my_top --clock-ns 1.5 --out runs/my
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chipflow import designs, layout, pdk, place, power, route, sim, size, sta, synth


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _summary(t):
    return {k: v for k, v in t.items() if k not in ("critical_path", "slack_by_endpoint", "net_slack")}


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
    rep = {"design": top, "params": params, "clock_ns": a.clock_ns}

    log(f"[1/7] synthesis: {top} {params}")
    net, rep["synth"] = synth.synthesize(sources, top, params, out)
    log(f"      {rep['synth']['cells']} cells, {rep['synth']['area_um2']:.0f} um2, {rep['synth']['flip_flops']} flip-flops, "
        f"{len(rep['synth']['lint_warnings'])} lint warnings")

    log("[2/7] gate-level simulation")
    s = sim.Simulator(net, lib)
    rng = np.random.default_rng(a.seed)
    if spec and spec.get("tb"):
        rep["verify"] = spec["tb"](s, params, rng)
        log(f"      golden model: {rep['verify']['mismatches']} mismatches in {rep['verify']['checked']} checked outputs")
    else:
        for _ in range(32):
            s.step({p: rng.integers(0, 2 ** 63, size=len(ns), dtype=np.uint64)
                    for p, (d, ns) in net.ports.items() if d == "input"})
        rep["verify"] = {"checked": 0, "mismatches": None, "cycles": s.cycles, "note": "random stimulus, activity only"}
        log("      no golden model: random stimulus for switching activity only")

    log("[3/7] floorplan + global placement + legalization")
    pl = place.place(net, lib, lef, util=a.util, seed=a.seed, log=log)
    rep["place"] = {k: v for k, v in pl.stats.items() if k != "gp_hist"}
    log(f"      core {pl.stats['core_um']} um, HPWL {pl.stats['hpwl_um']:.0f} um, legalization moved {pl.stats['legal_disp_avg_um']} um avg")

    log("[4/7] timing after placement")
    t_pl = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"])
    weights = None
    if a.timing_driven:  # weight near-critical nets, re-place from the current solution
        weights = place.timing_weights(t_pl, a.clock_ns)
        log(f"      fmax {t_pl['fmax_mhz']} MHz; timing-driven re-placement weighting {len(weights)} nets")
        pl = place.place(net, lib, lef, util=a.util, seed=a.seed, init=np.stack([pl.x + pl.w / 2, pl.y + pl.h / 2], 1), net_weight=weights)
        t_pl = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"])
        rep["place"] = {k: v for k, v in pl.stats.items() if k != "gp_hist"} | {"timing_driven_nets": len(weights)}
    if a.size and t_pl["wns_ns"] < 0:
        rep["sizing"] = size.upsize(net, lib, lef, pl, a.clock_ns, log=log)
        log(f"      gate sizing: {rep['sizing']['cells_upsized']} cells upsized, WNS {rep['sizing']['wns_before_ns']:+.3f} -> {rep['sizing']['wns_after_ns']:+.3f} ns")
        t_pl = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"])
    rep["sta_placed"] = _summary(t_pl)
    log(f"      WNS {rep['sta_placed']['wns_ns']:+.3f} ns, fmax {rep['sta_placed']['fmax_mhz']} MHz")

    log("[5/7] global routing")
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

    log("[6/7] timing + power after routing")
    t_rt = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"], rt["net_len"])
    if a.size and t_rt["wns_ns"] < 0:  # post-route repair on routed lengths, then reroute the moved cells
        rep["sizing_post_route"] = size.upsize(net, lib, lef, pl, a.clock_ns, log=log, net_len=rt["net_len"])
        rt = route.route(net, lib, pl, lef)
        rep["route"] |= {k: v for k, v in rt.items() if k not in ("net_len", "grid_obj", "conns")}
        t_rt = sta.analyze(net, lib, a.clock_ns, pl, lef["macros"], rt["net_len"])
        log(f"      post-route sizing: {rep['sizing_post_route']['cells_upsized']} cells, rerouted overflow {rt['overflow']}")
    rep["sta_routed"] = _summary(t_rt)
    rep["critical_path"] = [{k: v for k, v in st.items() if k != "idx"} for st in t_rt["critical_path"]]
    rep["power"] = power.analyze(net, lib, L["voltage"], a.clock_ns, s.toggles, s.cycles, 64, rt["net_len"])
    log(f"      WNS {t_rt['wns_ns']:+.3f} ns, TNS {t_rt['tns_ns']} ns, fmax {t_rt['fmax_mhz']} MHz, "
        f"{len(t_rt['critical_path'])} stages; power {rep['power']['total_mw']} mW")

    log("[7/7] outputs")
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
    f.add_argument("--clock-ns", type=float, default=2.0)
    f.add_argument("--util", type=float, default=0.6)
    f.add_argument("--routability-rounds", type=int, default=2, help="re-place congested regions up to N times")
    f.add_argument("--timing-driven", action=argparse.BooleanOptionalAction, default=True, help="re-place with critical nets weighted")
    f.add_argument("--size", action=argparse.BooleanOptionalAction, default=True, help="upsize cells when timing fails")
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
