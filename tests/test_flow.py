import collections
import itertools

import numpy as np
import pytest

from chipflow import designs, pdk, place, route, sim, size, sta, synth
from chipflow.pdk import Table
from chipflow.sim import compile_function


@pytest.fixture(scope="module")
def lib():
    pdk.fetch()
    return pdk.load_liberty()["cells"]


@pytest.fixture(scope="module")
def sa2(lib, tmp_path_factory):
    net, rep = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", {"N": 2},
                                tmp_path_factory.mktemp("sa2"))
    return net, rep


@pytest.mark.parametrize("expr, names, ref", [
    ("!(A1 & A2)", "A1 A2", lambda a, b: not (a and b)),
    ("((A & B) | (CI & (A | B)))", "A B CI", lambda a, b, c: (a and b) or (c and (a or b))),
    ("(CI ^ (A ^ B))", "A B CI", lambda a, b, c: c ^ a ^ b),
    ("((S & B) | (A & !S))", "A B S", lambda a, b, s: (s and b) or (a and not s)),
    ("A B + C'", "A B C", lambda a, b, c: (a and b) or not c),        # juxtaposition AND, postfix NOT
    ("!(((C1 & C2) | B) | A)", "A B C1 C2", lambda a, b, c1, c2: not ((c1 and c2) or b or a)),
])
def test_liberty_function_compiles_to_correct_truth_table(expr, names, ref):
    fn = eval(f"lambda v, Z, O: {compile_function(expr)}")  # compile_function emits only pin lookups and ops
    for bits in itertools.product([0, 1], repeat=len(names.split())):
        words = {k: sim.ONES if v else np.uint64(0) for k, v in zip(names.split(), bits)}
        assert int(fn(words, np.uint64(0), sim.ONES) & np.uint64(1)) == int(bool(ref(*bits))), (expr, bits)


def test_nldm_table_interpolates_bilinearly():
    t = Table([0.0, 1.0], [0.0, 2.0], [[0.0, 2.0], [1.0, 3.0]])
    assert t(0.5, 1.0) == pytest.approx(1.5) and t(2.0, 0.0) == pytest.approx(2.0)  # linear extrapolation


def test_systolic_gate_level_matches_golden_matmul(lib, sa2):
    net, rep = sa2
    assert rep["flip_flops"] == 2 * 2 * (8 + 8 + 24) and not rep["lint_warnings"]
    res = designs.systolic_tb(sim.Simulator(net, lib), {"N": 2}, np.random.default_rng(0))
    assert res["checked"] == 256 and res["mismatches"] == 0


def test_simt_gate_level_matches_golden_isa(lib, tmp_path):
    p = {"LANES": 2, "WARPS": 2, "REGS": 8, "W": 16}
    net, _ = synth.synthesize([designs.DESIGN_DIR / "simt_core.v"], "simt_core", p, tmp_path)
    res = designs.simt_tb(sim.Simulator(net, lib), p, np.random.default_rng(3), cycles=60)
    assert res["checked"] > 5000 and res["mismatches"] == 0


def test_backend_is_legal_connected_and_timed(lib, sa2):
    net, _ = sa2
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    for row in np.unique(pl.y):                      # no overlaps, inside the core, on the site grid
        i = np.where(pl.y == row)[0]
        o = np.argsort(pl.x[i])
        assert np.all(pl.x[i][o][1:] >= pl.x[i][o][:-1] + pl.w[i][o][:-1] - 1e-6)
    assert (pl.x >= pl.core[0] - 1e-9).all() and (pl.x + pl.w <= pl.core[2] + 1e-9).all()
    sites = (pl.x - pl.core[0]) / pl.site_w
    assert np.allclose(sites, np.round(sites), atol=1e-6)
    rt = route.route(net, lib, pl, lef)
    assert rt["overflow"] == 0
    for _, a, b, path in rt["conns"]:                # every 2-pin connection is one connected path
        adj = collections.defaultdict(set)
        for k, y, x in path:
            u, v = ((x, y), (x + 1, y)) if k == "h" else ((x, y), (x, y + 1))
            adj[u].add(v)
            adj[v].add(u)
        seen, todo = {a}, [a]
        while todo:
            for v in adj[todo.pop()] - seen:
                seen.add(v)
                todo.append(v)
        assert b in seen
    t = sta.analyze(net, lib, 2.0, pl, lef["macros"], rt["net_len"])
    path = t["critical_path"]
    arr = [s["arrival_ns"] for s in path]
    assert arr == sorted(arr) and path[0]["cell"].startswith("DFF")   # launched by a flip-flop, monotone
    for s0, s1 in zip(path, path[1:]):
        assert s1["arrival_ns"] == pytest.approx(s0["arrival_ns"] + s1["incr_ns"] + s1["wire_ns"], abs=2e-4)
    assert 0 < t["fmax_mhz"] < 2000
    assert min(t["slack_by_endpoint"].values()) == t["wns_ns"]   # each endpoint keeps its worse edge
    assert min(t["net_slack"].values()) == pytest.approx(t["wns_ns"], abs=5e-5)   # backward pass agrees (wns rounded to 1e-4)


def test_timing_repair_upsizes_without_changing_logic(lib, tmp_path):
    net, _ = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", {"N": 2}, tmp_path)
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    t = sta.analyze(net, lib, 1.5, pl, lef["macros"])
    w = place.timing_weights(t, 1.5)
    assert w and all(1.0 < v <= 5.0 for v in w.values()) and len(w) < net.n_nets
    before = list(net.cells)
    r = size.upsize(net, lib, lef, pl, 1.5)
    assert r["cells_upsized"] > 0 and r["wns_after_ns"] > r["wns_before_ns"] and r["area_after_um2"] > r["area_before_um2"]
    for (_, a), (_, b) in zip(before, net.cells):     # only drive strength changed, same family
        assert a.rsplit("_X", 1)[0] == b.rsplit("_X", 1)[0]
    res = designs.systolic_tb(sim.Simulator(net, lib), {"N": 2}, np.random.default_rng(0))
    assert res["mismatches"] == 0                     # sized netlist still computes the same matmul


def test_clock_tree_buffering_and_hold_fix(lib, tmp_path):
    from chipflow import buffer
    net, _ = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", {"N": 2}, tmp_path)
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    buffer.cts(net, lib, lef, pl)
    r = buffer.repair_design(net, lib, lef, pl, 2.0)
    t = sta.analyze(net, lib, 2.0, pl, lef["macros"])
    assert t["clock"]["propagated"] and t["clock"]["skew_ns"] < 0.05 and t["hold_failing"] == 0
    assert r["violations_after"] == {"slew": 0, "cap": 0}
    _, sinks = net.drivers_sinks(lib)
    assert max(len(s) for n, s in sinks.items() if n not in {k for p in net.ports.values() for k in p[1]}) <= 16
    c = min(t["hold_by_ff"], key=t["hold_by_ff"].get)   # skew the tightest flip-flop late: 9 x ~25 ps beats ~+0.13 ns of hold slack
    ck = lib[net.cells[c][1]].ff["clock"]
    for k in range(9):
        out = buffer._new_net(net)
        x, y = pl.pin_xy(c, ck, lef["macros"], net.cells[c][1])
        buffer._add_cell(net, pl, lef, f"skew_{k}", "CLKBUF_X1", {"A": net.pins[c][ck], "Z": out}, x, y)
        net.pins[c][ck] = out
    buffer._legalize(pl, lef)
    assert sta.analyze(net, lib, 2.0, pl, lef["macros"])["hold_failing"] >= 1
    h = buffer.fix_hold(net, lib, lef, pl, 2.0)
    assert h["buffers"] > 0 and h["hold_failing"] == 0 and h["whs_after_ns"] >= 0
    res = designs.systolic_tb(sim.Simulator(net, lib), {"N": 2}, np.random.default_rng(0))
    assert res["mismatches"] == 0                     # clock tree, buffers and delay cells keep the logic


@pytest.mark.parametrize("reg_in", [0, 1])
def test_io_constraints_time_the_ports(lib, tmp_path, reg_in):
    p = {"N": 2, "REG_IN": reg_in}
    net, _ = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", p, tmp_path)
    assert designs.systolic_tb(sim.Simulator(net, lib), p, np.random.default_rng(0))["mismatches"] == 0
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    free = sta.analyze(net, lib, 2.0, pl, lef["macros"])
    net.io = {"input_delay": 0.6, "output_delay": 0.6, "driver": "BUF_X4", "output_load": 4.0}
    t = sta.analyze(net, lib, 2.0, pl, lef["macros"])
    assert t["wns_ns"] <= free["wns_ns"] + 1e-9            # a budget never makes timing better
    starts_at_input = t["critical_path"][0]["point"] == "input"
    assert starts_at_input == (reg_in == 0)                 # registered inputs take the port off the critical path
    assert t["whs_ns"] is not None and t["whs_ns"] > 0      # inputs now hold-checked; 0.6 ns delay is ample
