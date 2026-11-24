"""Export both benchmark designs, run through the whole flow, for the static explorer page.

  python demo/build_data.py        # writes demo/data.json (~35 s on CPU)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chipflow import cli
from chipflow.layout import _group, congestion_map

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"systolic": {"clock_ns": 2.0, "label": "TPU-style 4×4 systolic array"},
        "simt": {"clock_ns": 3.0, "label": "GPU-style SIMT slice, 4 lanes × 2 warps"}}
r2 = lambda v: round(float(v), 2)


def export(key: str, spec: dict) -> dict:
    a = argparse.Namespace(design=key, verilog=None, top=None, param=None, clock_ns=spec["clock_ns"],
                           util=0.6, seed=0, routability_rounds=2, timing_driven=True, size=True, cts=True, buffer=True, io=True, io_budget=0.3, input_driver="BUF_X4", output_load=4.0, out=str(ROOT / "runs" / f"demo_{key}"))
    rep, ctx = cli.flow(a)
    net, lib, pl, rt, t = ctx["net"], ctx["lib"], ctx["pl"], ctx["route"], ctx["sta"]
    types = sorted({typ for _, typ in net.cells})
    tix = {typ: i for i, typ in enumerate(types)}
    groups, gix, g_of = [], {}, []
    for inst, _ in net.cells:
        k = _group(inst)
        label = ("PE " + ",".join(k) if k[1] else "lane " + k[0]) if k else "shared"
        if label not in gix:
            gix[label] = len(groups)
            groups.append(label)
        g_of.append(gix[label])
    util = congestion_map(rt["grid_obj"])
    ports = list(net.ports)
    slacks = np.array(list(t["slack_by_endpoint"].values()))
    return {
        "key": key, "label": spec["label"], "clock_ns": spec["clock_ns"], "params": ctx["params"],
        "die": [r2(v) for v in pl.die], "core": [r2(v) for v in pl.core], "row_h": pl.row_h,
        "types": [{"name": typ, "w": r2(ctx["lef"]["macros"][typ].width), "ff": bool(lib[typ].ff)} for typ in types],
        "groups": groups,
        "cells": {"x": [r2(v) for v in pl.x], "y": [r2(v) for v in pl.y],
                  "t": [tix[typ] for _, typ in net.cells], "g": g_of},
        "io": {"ports": ports, "pins": [[r2(x), r2(y), ports.index(p)] for (p, b), (x, y) in pl.io.items()]},
        "congestion": {"nx": int(util.shape[1]), "ny": int(util.shape[0]), "gcell": r2(rt["gcell_um"]),
                       "util": [round(float(v), 2) for v in util.ravel()]},
        "critical": {"endpoint": t["critical_endpoint"], "wns_ns": t["wns_ns"], "sink_idx": t["critical_sink_idx"],
                     "stages": [{k: (v if not isinstance(v, (np.generic,)) else v.item()) for k, v in st.items()}
                                for st in t["critical_path"]]},
        "slacks": [round(float(s), 3) for s in np.sort(slacks)],
        "report": {k: rep[k] for k in ("io", "synth", "verify", "verify_backend", "place", "cts", "repair", "sizing", "hold_fix", "sta_placed", "sta_routed", "route", "power", "seconds") if k in rep},
        "verilog": Path(ctx["sources"][0]).read_text(),
    }


def main():
    data = {"designs": [export(k, v) for k, v in RUNS.items()]}
    for d in data["designs"]:
        d["report"]["synth"].pop("lint_warnings", None)
    (ROOT / "demo" / "data.json").write_text(json.dumps(cli._clean(data), separators=(",", ":")))


if __name__ == "__main__":
    main()
