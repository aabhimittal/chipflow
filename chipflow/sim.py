"""Bit-parallel gate-level simulation of the synthesized netlist.

Each net holds a uint64 word: bit k is the value in independent test lane k, so one pass
simulates 64 stimuli. Cell behavior comes from the Liberty `function` strings, compiled once
per cell type. Also records per-net toggle counts for power analysis.
"""
from __future__ import annotations

import re

import numpy as np

ONES = np.uint64(0xFFFFFFFFFFFFFFFF)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def compile_function(expr: str) -> str:
    """Liberty boolean expression -> Python bitwise expression over a dict `v` of pin words.
    Precedence: ! / ' (postfix) > & * juxtaposition > ^ > | +."""
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[01]|[!'&*|+^()]", expr)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def primary():
        t = take()
        if t == "(":
            e = or_()
            assert take() == ")", expr
        elif t == "!":
            e = f"(~{primary()})"
        elif t in ("0", "1"):
            e = "Z" if t == "0" else "O"
        elif _IDENT.fullmatch(t):
            e = f"v[{t!r}]"
        else:
            raise ValueError(f"bad liberty function {expr!r}")
        while peek() == "'":
            take()
            e = f"(~{e})"
        return e

    def and_():
        e = primary()
        while peek() in ("&", "*") or (peek() is not None and (peek() in ("(", "!", "0", "1") or _IDENT.fullmatch(peek()))):
            if peek() in ("&", "*"):
                take()
            e = f"({e} & {primary()})"
        return e

    def xor_():
        e = and_()
        while peek() == "^":
            take()
            e = f"({e} ^ {and_()})"
        return e

    def or_():
        e = xor_()
        while peek() in ("|", "+"):
            take()
            e = f"({e} | {xor_()})"
        return e

    out = or_()
    if pos != len(toks):
        raise ValueError(f"trailing tokens in {expr!r}")
    return out


def comb_order(net, lib) -> list:
    """Combinational cells in topological order (flip-flops cut the cycles); iterative DFS."""
    drv = {}
    for c, (_, t) in enumerate(net.cells):
        if not lib[t].ff:
            for p in lib[t].outputs():
                drv[net.pins[c].get(p.name)] = c
    order, state = [], {}
    for c0, (_, t) in enumerate(net.cells):
        if lib[t].ff or c0 in state:
            continue
        stack = [(c0, iter(net.pins[c0].items()))]
        state[c0] = 1
        while stack:
            c, it = stack[-1]
            for p, n in it:
                if lib[net.cells[c][1]].pins[p].direction == "output":
                    continue
                d = drv.get(n)
                if d is None or state.get(d) == 2:
                    continue
                if state.get(d) == 1:
                    raise ValueError("combinational loop through " + net.cells[d][0])
                state[d] = 1
                stack.append((d, iter(net.pins[d].items())))
                break
            else:
                state[c] = 2
                order.append(c)
                stack.pop()
    return order


class Simulator:
    def __init__(self, net, lib):
        self.net, self.lib = net, lib
        z = np.uint64(0)
        self.Z, self.O = z, ONES
        self.fns = {}
        for typ in {t for _, t in net.cells}:
            cell = lib[typ]
            outs = {}
            for p in cell.outputs():
                f = p.function
                if cell.ff and f in (cell.ff["state"], cell.ff["state_n"]):
                    continue  # flip-flop outputs come from state
                if f is None:
                    raise ValueError(f"{typ}.{p.name} has no function; unsupported cell")
                outs[p.name] = eval(f"lambda v, Z, O: {compile_function(f)}")  # expr built only from pin idents
            self.fns[typ] = outs
        self.ffs = [c for c, (_, t) in enumerate(net.cells) if lib[t].ff]
        self.order = comb_order(net, lib)
        self.v = np.zeros(net.n_nets, dtype=np.uint64)
        for n, b in net.const.items():
            self.v[n] = ONES if b else z
        self.state = np.zeros(len(self.ffs), dtype=np.uint64)
        self.toggles = np.zeros(net.n_nets, dtype=np.int64)
        self.cycles = 0

    def step(self, inputs: dict) -> dict:
        """One clock cycle. inputs: port -> array of uint64 words per bit (or an int for all lanes).
        Returns output port words sampled before the clock edge."""
        net, v = self.net, self.v
        prev = v.copy()
        for name, words in inputs.items():
            v[net.ports[name][1]] = words
        for i, c in enumerate(self.ffs):
            typ = net.cells[c][1]
            ff = self.lib[typ].ff
            for p, n in net.pins[c].items():
                f = self.lib[typ].pins[p].function
                if f == ff["state"]:
                    v[n] = self.state[i]
                elif f == ff["state_n"]:
                    v[n] = ~self.state[i]
        for c in self.order:
            typ = net.cells[c][1]
            pins = {p: v[n] for p, n in net.pins[c].items()}
            for p, fn in self.fns[typ].items():
                v[net.pins[c][p]] = fn(pins, self.Z, self.O)
        out = {name: v[nets].copy() for name, (d, nets) in net.ports.items() if d == "output"}
        for i, c in enumerate(self.ffs):  # rising edge: capture D
            self.state[i] = v[net.pins[c][self.lib[net.cells[c][1]].ff["next"]]]
        self.toggles += np.bitwise_count(prev ^ v).astype(np.int64)
        self.cycles += 1
        return out


def pack(values, width: int) -> np.ndarray:
    """values[lane] (ints, up to 64 lanes) -> per-bit uint64 lane words, LSB first."""
    words = np.zeros(width, dtype=np.uint64)
    for lane, x in enumerate(values):
        for b in range(width):
            if (int(x) >> b) & 1:
                words[b] |= np.uint64(1) << np.uint64(lane)
    return words


def unpack(words, lanes: int, signed: bool = False) -> list:
    w = len(words)
    out = []
    for lane in range(lanes):
        x = sum(((int(words[b]) >> lane) & 1) << b for b in range(w))
        out.append(x - (1 << w) if signed and x >> (w - 1) else x)
    return out
