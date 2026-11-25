"""Front end: Verilog -> lint -> Yosys synthesis onto Nangate45 -> flat gate netlist."""
from __future__ import annotations

import dataclasses as dc
import json
import re
import shutil
import subprocess
from pathlib import Path

from chipflow import pdk


@dc.dataclass
class Netlist:
    top: str
    cells: list            # [(instance name, cell type)]
    pins: list             # per cell: {pin: net id}
    ports: dict            # name -> (direction, [net ids])
    n_nets: int
    const: dict            # net id -> 0/1 for tie nets
    io: dict | None = None # timing constraints in the form chipflow.sdc builds (per port, exceptions)
    names: dict = dc.field(default_factory=dict)  # net id -> RTL wire names (e.g. "row[0].col[1].pe.acc[3]")

    def drivers_sinks(self, lib):
        """drv[net] = (cell, pin) or ('port', name, bit); sinks[net] = [(cell, pin) or ('port', name, bit)]."""
        drv, sinks = {}, {n: [] for n in range(self.n_nets)}
        for c, (_, typ) in enumerate(self.cells):
            for p, n in self.pins[c].items():
                if lib[typ].pins[p].direction == "output":
                    drv[n] = (c, p)
                else:
                    sinks[n].append((c, p))
        for name, (d, nets) in self.ports.items():
            for b, n in enumerate(nets):
                (sinks[n].append(("port", name, b)) if d == "output" else drv.__setitem__(n, ("port", name, b)))
        return drv, sinks


def synthesize(sources, top: str, params: dict | None = None, workdir: str | Path = ".") -> tuple[Netlist, dict]:
    """Run Yosys (YoWASP build) and return the flat netlist plus a front-end report."""
    wd = Path(workdir).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    # YoWASP's WASI sandbox only sees its working directory: stage every input there
    lib = "cells.lib"
    shutil.copyfile(pdk.pdk_dir() / "NangateOpenCellLibrary_typical.lib", wd / lib)
    srcs = []
    for i, s in enumerate(sources):
        srcs.append(f"src{i}_{Path(s).name}")
        shutil.copyfile(s, wd / srcs[-1])
    chp = "".join(f"chparam -set {k} {int(v)} {top}\n" for k, v in (params or {}).items())
    script = (
        "".join(f"read_verilog -sv {s}\n" for s in srcs) + chp +
        f"hierarchy -check -top {top}\nrename -top {top}\ncheck\n"   # chparam before hierarchy dodges a 0.71 assert
        f"synth -top {top}\ndfflibmap -liberty {lib}\nabc -liberty {lib}\n"
        f"flatten\nopt_clean\nsetundef -zero\nhilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z\n"  # no -purge: SDC matches the RTL wire names
        f"stat -liberty {lib}\nwrite_json netlist.json\nwrite_verilog -noattr netlist.v\n")
    (wd / "synth.ys").write_text(script)
    r = subprocess.run(["yowasp-yosys", "-q", "-s", "synth.ys", "-l", "synth.log"], cwd=wd,
                       capture_output=True, text=True)
    log = (wd / "synth.log").read_text() if (wd / "synth.log").exists() else r.stderr
    if r.returncode:
        err = [l for l in log.splitlines() if "ERROR" in l]
        raise RuntimeError(f"yosys failed: {err[-1] if err else r.stderr[-500:]}")
    warnings = sorted({l.strip() for l in log.splitlines() if l.strip().startswith("Warning:")})
    area = re.findall(r"Chip area for module .*?:\s*([0-9.]+)", log)
    net = load_json(wd / "netlist.json", top)
    report = {"cells": len(net.cells), "area_um2": float(area[-1]) if area else None,
              "lint_warnings": warnings, "flip_flops": sum(t.startswith("DFF") for _, t in net.cells)}
    return net, report


def load_json(path, top: str) -> Netlist:
    m = json.loads(Path(path).read_text())["modules"][top]
    ids: dict = {}
    const = {}

    def nid(b):
        if b in ("0", "1", "x", "z"):
            key = "c" + ("1" if b == "1" else "0")
            if key not in ids:
                ids[key] = len(ids)
                const[ids[key]] = int(b == "1")
            return ids[key]
        return ids.setdefault(b, len(ids))

    cells, pins = [], []
    for name, c in m["cells"].items():
        if c["type"].startswith("$"):  # e.g. $scopeinfo left by flatten
            continue
        cells.append((name, c["type"]))
        pins.append({p: nid(bits[0]) for p, bits in c["connections"].items()})
    ports = {name: (p["direction"], [nid(b) for b in p["bits"]]) for name, p in m["ports"].items()}
    names: dict = {}
    for w, v in m["netnames"].items():
        if not v["hide_name"]:
            for i, b in enumerate(v["bits"]):
                if b not in ("0", "1", "x", "z") and b in ids:
                    names.setdefault(ids[b], []).append(f"{w}[{i}]" if len(v["bits"]) > 1 else w)
    return Netlist(top, cells, pins, ports, len(ids), const, names=names)
