"""Nangate45 open cell library: fetch (pinned), Liberty timing/power parser, LEF geometry parser.

Nangate45 (Apache-2.0, purposely non-manufacturable) is fetched from OpenROAD-flow-scripts at a
pinned commit and checked against SHA-256, not vendored.
"""
from __future__ import annotations

import bisect
import dataclasses as dc
import hashlib
import os
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

ORFS = ("https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/"
        "f5504444e08f756336ed39935baafaaa98d45374/flow/platforms/nangate45")  # pinned commit
FILES = {
    "NangateOpenCellLibrary_typical.lib": ("lib", "8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1"),
    "NangateOpenCellLibrary.tech.lef": ("lef", "834a79295054cd4209178d1bade67c353863c47bb4b3c22ee38b862b7cec37f2"),
    "NangateOpenCellLibrary.macro.mod.lef": ("lef", "a43aea339f12a57a63497783e508ba16f3da2dc056d3247dec7d99707c2dedef"),
}
# Wire RC of metal3 from ORFS nangate45/setRC.tcl, in ohm/um and fF/um.
WIRE_R_OHM_UM, WIRE_C_FF_UM = 3.57147, 0.103981
ROUTING_LAYERS = ("metal2", "metal3", "metal4", "metal5", "metal6")


def pdk_dir() -> Path:
    return Path(os.environ.get("CHIPFLOW_PDK", Path.home() / ".cache" / "chipflow" / "nangate45"))


def fetch(force: bool = False) -> Path:
    d = pdk_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name, (sub, sha) in FILES.items():
        p = d / name
        if p.exists() and not force and hashlib.sha256(p.read_bytes()).hexdigest() == sha:
            continue
        data = urllib.request.urlopen(f"{ORFS}/{sub}/{name}", timeout=60).read()
        if hashlib.sha256(data).hexdigest() != sha:  # trust boundary: refuse a changed upstream file
            raise RuntimeError(f"{name}: checksum mismatch; upstream changed, update the pin in pdk.py")
        p.write_bytes(data)
    return d


# ---------------------------------------------------------------- Liberty

_TOK = re.compile(r'"(?:[^"\\]|\\.)*"|[A-Za-z0-9_.\-+!\[\]\']+|[(){}:;,]')


@dc.dataclass
class Group:
    kind: str
    args: list
    attrs: dict
    groups: list

    def sub(self, kind):
        return [g for g in self.groups if g.kind == kind]


def parse_liberty_text(text: str) -> Group:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S).replace("\\\n", " ")
    toks = _TOK.findall(text)
    pos = 0

    def val(t):
        return t[1:-1] if t.startswith('"') else t

    def group_body(g):
        nonlocal pos
        while toks[pos] != "}":
            name = toks[pos]
            pos += 1
            if toks[pos] == ":":
                g.attrs[name] = val(toks[pos + 1])
                pos += 2
                if toks[pos] == ";":
                    pos += 1
            elif toks[pos] == "(":
                pos += 1
                args = []
                while toks[pos] != ")":
                    if toks[pos] != ",":
                        args.append(val(toks[pos]))
                    pos += 1
                pos += 1
                if toks[pos] == "{":
                    pos += 1
                    child = Group(name, args, {}, [])
                    group_body(child)
                    g.groups.append(child)
                else:  # complex attribute, e.g. index_1 ("0.1, 0.2")
                    g.attrs.setdefault(name, args)
                    if toks[pos] == ";":
                        pos += 1
            else:
                raise ValueError(f"liberty parse error near token {pos}: {toks[pos - 3:pos + 3]}")
        pos += 1

    assert toks[0] == "library"
    root = Group("library", [val(toks[2])], {}, [])
    pos = 5  # library ( name ) {
    group_body(root)
    return root


def _floats(args) -> list[float]:
    return [float(v) for a in args for v in a.replace(",", " ").split()]


@dc.dataclass
class Table:
    """NLDM lookup table, bilinear inter/extrapolation. x = variable_1, y = variable_2."""
    x: list
    y: list
    v: list  # rows over x, columns over y

    def __call__(self, a: float, b: float = 0.0) -> float:
        def seg(axis, q):
            if len(axis) == 1:
                return 0, 0, 0.0
            i = min(max(bisect.bisect_right(axis, q) - 1, 0), len(axis) - 2)
            return i, i + 1, (q - axis[i]) / (axis[i + 1] - axis[i])
        i0, i1, tx = seg(self.x, a)
        if not self.y or len(self.y) == 1:
            return self.v[i0][0] + tx * (self.v[i1][0] - self.v[i0][0])
        j0, j1, ty = seg(self.y, b)
        v0 = self.v[i0][j0] + ty * (self.v[i0][j1] - self.v[i0][j0])
        v1 = self.v[i1][j0] + ty * (self.v[i1][j1] - self.v[i1][j0])
        return v0 + tx * (v1 - v0)


@dc.dataclass
class Arc:
    related: str
    kind: str          # combinational | rising_edge | setup_rising | hold_rising ...
    sense: str         # positive_unate | negative_unate | non_unate
    tables: dict       # cell_rise/cell_fall/rise_transition/fall_transition/rise_constraint/fall_constraint


@dc.dataclass
class Pin:
    name: str
    direction: str
    cap: float = 0.0
    function: str | None = None
    clock: bool = False
    arcs: list = dc.field(default_factory=list)
    max_cap: float = float("inf")


@dc.dataclass
class Cell:
    name: str
    area: float
    leakage_nw: float
    pins: dict
    ff: dict | None = None  # {"state": "IQ", "next": "D", "clock": "CK"}

    def outputs(self):
        return [p for p in self.pins.values() if p.direction == "output"]


def _table(g: Group, templates: dict) -> Table:
    t = templates.get(g.args[0] if g.args else "", {})
    x = _floats(g.attrs["index_1"]) if "index_1" in g.attrs else t.get("index_1", [0.0])
    y = _floats(g.attrs["index_2"]) if "index_2" in g.attrs else t.get("index_2", [])
    flat = _floats(g.attrs["values"])
    ny = max(len(y), 1)
    if t.get("swap"):  # template lists capacitance first: store as (slew, cap) anyway
        x, y = y, x
        rows = [flat[i * len(x):(i + 1) * len(x)] for i in range(len(y))]
        return Table(x, y, [list(r) for r in zip(*rows)])
    return Table(x, y, [flat[i * ny:(i + 1) * ny] for i in range(len(x))])


@lru_cache(maxsize=4)
def load_liberty(path: str | None = None) -> dict:
    root = parse_liberty_text(Path(path or pdk_dir() / "NangateOpenCellLibrary_typical.lib").read_text())
    templates = {}
    for t in root.sub("lu_table_template"):
        info = {k: _floats(t.attrs[k]) for k in ("index_1", "index_2") if k in t.attrs}
        info["swap"] = t.attrs.get("variable_1", "").endswith("capacitance")
        templates[t.args[0]] = info
    cells = {}
    for c in root.sub("cell"):
        ffg = c.sub("ff")
        ff = None
        if ffg:
            ff = {"state": ffg[0].args[0], "state_n": ffg[0].args[1], "next": ffg[0].attrs["next_state"],
                  "clock": ffg[0].attrs["clocked_on"]}
        pins = {}
        for p in c.sub("pin"):
            pin = Pin(p.args[0], p.attrs.get("direction", "input"), float(p.attrs.get("capacitance", 0.0)),
                      p.attrs.get("function"), p.attrs.get("clock") == "true",
                      max_cap=float(p.attrs.get("max_capacitance", "inf")))
            for t in p.sub("timing"):
                tabs = {g.kind: _table(g, templates) for g in t.groups
                        if g.kind in ("cell_rise", "cell_fall", "rise_transition", "fall_transition",
                                      "rise_constraint", "fall_constraint")}
                pin.arcs.append(Arc(t.attrs.get("related_pin", ""), t.attrs.get("timing_type", "combinational"),
                                    t.attrs.get("timing_sense", "non_unate"), tabs))
            pins[pin.name] = pin
        cells[c.args[0]] = Cell(c.args[0], float(c.attrs.get("area", 0.0)),
                                float(c.attrs.get("cell_leakage_power", 0.0)), pins, ff)
    return {"cells": cells, "voltage": float(root.attrs.get("nom_voltage", 1.1))}


# ---------------------------------------------------------------- LEF

@dc.dataclass
class Macro:
    name: str
    width: float
    height: float
    pins: dict  # pin -> (dx, dy) center of its first port rectangle


@lru_cache(maxsize=4)
def load_lef(tech: str | None = None, macros: str | None = None) -> dict:
    d = pdk_dir()
    layers, site = {}, None
    cur = None
    for line in Path(tech or d / "NangateOpenCellLibrary.tech.lef").read_text().splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "LAYER" and len(w) == 2:
            cur = layers.setdefault(w[1], {})
        elif cur is not None and w[0] in ("PITCH", "DIRECTION") and len(w) >= 2:
            cur[w[0].lower()] = float(w[1]) if w[0] == "PITCH" else w[1]
        elif w[0] == "SITE":
            cur = None
        elif w[0] == "SIZE" and site is None:
            site = (float(w[1]), float(w[3]))
    out, m, pin, mpin = {}, None, None, False
    for line in Path(macros or d / "NangateOpenCellLibrary.macro.mod.lef").read_text().splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "MACRO":
            m = Macro(w[1], 0.0, 0.0, {})
            out[w[1]] = m
        elif m and w[0] == "SIZE":
            m.width, m.height = float(w[1]), float(w[3])
        elif m and w[0] == "PIN":
            pin, mpin = w[1], True
        elif m and w[0] == "RECT" and mpin and pin not in m.pins:
            x0, y0, x1, y1 = map(float, w[1:5])
            m.pins[pin] = ((x0 + x1) / 2, (y0 + y1) / 2)
        elif w[0] == "END" and len(w) > 1 and w[1] == pin:
            mpin = False
    return {"layers": {k: v for k, v in layers.items() if "pitch" in v}, "site": site, "macros": out}
