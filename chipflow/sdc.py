"""Timing constraints: a reader for the common subset of SDC, and the flow's default I/O budget.

Both produce one dict on `net.io`, which chipflow.sta reads:

  input_delay  {port: (max, min) ns}    arrival of the outside world at an input (setup uses max, hold min)
  output_delay {port: (max, min) ns}    time the outside world needs after an output
  driver       {port: lib cell}         cell assumed to drive an input (none: ideal edge)
  output_load  {port: fF}               load on an output (none: 0)
  false_from   {start}                  paths from these start points are not timed
  false_to     {end}                    paths into these end points are not timed
  multicycle   {end: (setup N, hold M)} capture N cycles later; hold edge (N - 1 - M) cycles after launch
  uncertainty  (setup, hold) ns         subtracted from every setup / hold slack

Start and end points are flip-flop indices or ("port", name). Ports without a delay are
unconstrained, as in SDC: paths from or to them are not timed.

Supported commands: create_clock, set_input_delay, set_output_delay, set_driving_cell, set_load,
set_false_path (-from or -to), set_multicycle_path (-to), set_clock_uncertainty, set and expr;
object queries get_ports, get_cells, get_clocks, all_inputs, all_outputs, all_registers,
remove_from_collection. Anything else is an error rather than silently ignored.
"""
from __future__ import annotations

import ast
import operator
import re
from pathlib import Path


def empty() -> dict:
    return {"input_delay": {}, "output_delay": {}, "driver": {}, "output_load": {}, "false_from": set(),
            "false_to": set(), "multicycle": {}, "uncertainty": (0.0, 0.0)}


def budget(net, delay_ns: float, driver: str, load_ff: float) -> dict:
    """Every input arrives delay_ns late through `driver`; every output must be ready delay_ns early."""
    io = empty()
    for name, (d, _) in net.ports.items():
        if d == "input":
            io["input_delay"][name] = (delay_ns, delay_ns)
            io["driver"][name] = driver
        else:
            io["output_delay"][name] = (delay_ns, delay_ns)
            io["output_load"][name] = load_ff
    return io


def summary(io: dict) -> dict:
    """JSON-friendly counts for the report."""
    return {"inputs_constrained": len(io["input_delay"]), "outputs_constrained": len(io["output_delay"]),
            "false_from": len(io["false_from"]), "false_to": len(io["false_to"]),
            "multicycle_endpoints": len(io["multicycle"]), "uncertainty_ns": list(io["uncertainty"])}


# --- Tcl, the small part SDC files use -------------------------------------------------------

def _words(src: str):
    """Split a script into commands of raw words: {braced}, "quoted", [bracketed] or bare."""
    src = src.replace("\\\n", " ")
    cmds, cur, i = [], [], 0
    while i < len(src):
        ch = src[i]
        if ch in " \t\r":
            i += 1
        elif ch in "\n;":
            if cur:
                cmds.append(cur)
            cur, i = [], i + 1
        elif ch == "#" and not cur:
            while i < len(src) and src[i] != "\n":
                i += 1
        elif ch in "{[":
            close, depth, j = "}" if ch == "{" else "]", 0, i
            while True:
                if j >= len(src):
                    raise ValueError(f"SDC: unbalanced {ch}")
                depth += src[j] == ch
                depth -= src[j] == close
                j += 1
                if depth == 0:
                    break
            cur.append(src[i:j])
            i = j
        elif ch == '"':
            j = src.index('"', i + 1)
            cur.append(src[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < len(src) and src[j] not in " \t\r\n;":
                j += 1
            cur.append(src[i:j])
            i = j
    if cur:
        cmds.append(cur)
    return cmds


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _arith(s: str) -> float:
    """expr: numbers and + - * / ( ) only; never Python eval."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"SDC: expr supports + - * / on numbers only: {s!r}")
    return float(ev(ast.parse(s, mode="eval")))


def _glob(pattern: str):
    # SDC patterns: * and ? are wildcards, brackets are literal bus bits (unlike fnmatch)
    return re.compile("".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern) + r"\Z")


class _Reader:
    def __init__(self, net, lib):
        self.net, self.lib, self.io, self.vars = net, lib, empty(), {}
        self.period, self.clock_ports, self.unc = None, [], [0.0, 0.0]
        self.ffs = [c for c, (_, t) in enumerate(net.cells) if lib[t].ff]

    # word -> value: str, float or list of objects ("port", name) / ("cell", idx) / ("clock", name)
    def value(self, w: str):
        if w.startswith("{"):
            return w[1:-1]
        if w.startswith("["):
            return self.run(_words(w[1:-1])[0])
        if re.fullmatch(r"\$\w+", w) and w[1:] in self.vars:
            return self.vars[w[1:]]   # keeps collections intact
        if w.startswith('"'):
            w = w[1:-1]
        return re.sub(r"\$(\w+)", lambda m: str(self.vars[m.group(1)]) if m.group(1) in self.vars else
                      _fail(f"SDC: unknown variable ${m.group(1)}"), w)

    def run(self, words):
        cmd, args = words[0], [self.value(w) for w in words[1:]]
        f = getattr(self, "c_" + cmd, None)
        if f is None:
            raise ValueError(f"SDC: unsupported command {cmd!r} (see chipflow/sdc.py for the supported subset)")
        return f(args)

    @staticmethod
    def opts(args, flags=(), keyed=()):
        """Split args into {-flag: True, -key: value} and positionals."""
        o, pos, i = {}, [], 0
        while i < len(args):
            a = args[i]
            if isinstance(a, str) and a in flags:
                o[a] = True
            elif isinstance(a, str) and a in keyed:
                o[a], i = args[i + 1], i + 1
            elif isinstance(a, str) and a.startswith("-") and not _is_num(a):
                raise ValueError(f"SDC: unsupported option {a}")
            else:
                pos.append(a)
            i += 1
        return o, pos

    def objs(self, v, kinds=("port", "cell")):
        if isinstance(v, str):     # a bare name list means ports, as in SDC
            v = self.c_get_ports([v])
        bad = [o for o in v if o[0] not in kinds]
        if bad:
            raise ValueError(f"SDC: expected {' or '.join(kinds)} objects, got {bad[0]}")
        return v

    # object queries
    def c_get_ports(self, args):
        out = []
        for pat in " ".join(a for a in args if not a.startswith("-")).split():
            hit = [("port", p) for p in self.net.ports if _glob(pat).match(p)]
            if not hit:
                raise ValueError(f"SDC: no port matches {pat!r}")  # shortcut: whole buses only, not a_west[3]
            out += hit
        return out

    def c_get_cells(self, args):
        """Flip-flops, matched by the RTL register bit they hold (row[0].col[1].pe.acc[3]) or instance name."""
        out = []
        for pat in " ".join(a for a in args if not a.startswith("-")).split():
            rx = _glob(pat)
            hit = [("cell", c) for c in self.ffs if rx.match(self.net.cells[c][0]) or any(
                rx.match(n) for p in self.lib[self.net.cells[c][1]].outputs()
                for n in self.net.names.get(self.net.pins[c].get(p.name), []))]
            if not hit:
                raise ValueError(f"SDC: no flip-flop matches {pat!r}")
            out += hit
        return out

    def c_get_clocks(self, args):
        return [("clock", a) for a in " ".join(args).split()]

    def c_all_inputs(self, args):
        return [("port", p) for p, (d, _) in self.net.ports.items() if d == "input"]

    def c_all_outputs(self, args):
        return [("port", p) for p, (d, _) in self.net.ports.items() if d == "output"]

    def c_all_registers(self, args):
        return [("cell", c) for c in self.ffs]

    def c_remove_from_collection(self, args):
        drop = set(args[1])
        return [o for o in args[0] if o not in drop]

    # variables
    def c_set(self, args):
        self.vars[args[0]] = args[1]
        return args[1]

    def c_expr(self, args):
        return _arith(" ".join(str(a) for a in args))

    # constraints
    def c_create_clock(self, args):
        o, pos = self.opts(args, keyed=("-period", "-name", "-waveform"))
        self.period = float(o["-period"])
        self.clock_ports = [p for _, p in self.objs(pos[0], ("port",))] if pos else []
        return [("clock", o.get("-name", self.clock_ports[0] if self.clock_ports else "clk"))]

    def _delay(self, args, key):
        o, pos = self.opts(args, flags=("-max", "-min", "-add_delay"), keyed=("-clock",))
        d = float(pos[0])
        for _, p in self.objs(pos[1], ("port",)):
            if p in self.clock_ports:
                continue
            mx, mn = self.io[key].get(p, (d, d))
            self.io[key][p] = (d if "-min" not in o else mx, d if "-max" not in o else mn)

    def c_set_input_delay(self, args):
        self._delay(args, "input_delay")

    def c_set_output_delay(self, args):
        self._delay(args, "output_delay")

    def c_set_driving_cell(self, args):
        o, pos = self.opts(args, keyed=("-lib_cell", "-pin"))
        if o["-lib_cell"] not in self.lib:
            raise ValueError(f"SDC: unknown library cell {o['-lib_cell']}")
        for _, p in self.objs(pos[0], ("port",)):
            self.io["driver"][p] = o["-lib_cell"]

    def c_set_load(self, args):
        _, pos = self.opts(args, flags=("-pin_load", "-wire_load"))
        for _, p in self.objs(pos[1], ("port",)):
            self.io["output_load"][p] = float(pos[0])

    @staticmethod
    def _key(o):
        return o[1] if o[0] == "cell" else ("port", o[1])

    def c_set_false_path(self, args):
        o, _ = self.opts(args, keyed=("-from", "-to"))
        if ("-from" in o) == ("-to" in o):  # one arrival per net: -from A -to B would need path tags
            raise ValueError("SDC: set_false_path needs exactly one of -from / -to")
        side = "-from" if "-from" in o else "-to"
        self.io["false_from" if side == "-from" else "false_to"] |= {self._key(x) for x in self.objs(o[side])}

    def c_set_multicycle_path(self, args):
        o, pos = self.opts(args, flags=("-setup", "-hold", "-end"), keyed=("-to", "-from"))
        if "-from" in o or "-to" not in o:
            raise ValueError("SDC: set_multicycle_path supports -to only")
        n = int(float(pos[0]))
        for x in self.objs(o["-to"]):
            s, h = self.io["multicycle"].get(self._key(x), (1, 0))
            self.io["multicycle"][self._key(x)] = (s, n) if "-hold" in o else (n, h)

    def c_set_clock_uncertainty(self, args):
        o, pos = self.opts(args, flags=("-setup", "-hold"))
        u = float(pos[0])
        if "-hold" not in o:
            self.unc[0] = u
        if "-setup" not in o:
            self.unc[1] = u


def _fail(msg):
    raise ValueError(msg)


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read(src, net, lib) -> tuple[dict, float | None]:
    """Parse an SDC file (path) or script (str with newlines). Returns (io dict, clock period or None)."""
    text = src if isinstance(src, str) and "\n" in src else Path(src).read_text()
    r = _Reader(net, lib)
    for words in _words(text):
        r.run(words)
    r.io["uncertainty"] = tuple(r.unc)
    return r.io, r.period
