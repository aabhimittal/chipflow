import collections
import itertools

import numpy as np
import pytest

from chipflow import designs, pdk, place, route, sdc, sim, size, sta, synth
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
    net.io = sdc.budget(net, 0.6, "BUF_X4", 4.0)
    t = sta.analyze(net, lib, 2.0, pl, lef["macros"])
    assert t["wns_ns"] <= free["wns_ns"] + 1e-9            # a budget never makes timing better
    starts_at_input = t["critical_path"][0]["point"] == "input"
    assert starts_at_input == (reg_in == 0)                 # registered inputs take the port off the critical path
    assert t["whs_ns"] is not None and t["whs_ns"] > 0      # inputs now hold-checked; 0.6 ns delay is ample


def test_sdc_reader_matches_budget_and_applies_exceptions(lib, tmp_path):
    p = {"N": 2, "REG_IN": 1}
    net, _ = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", p, tmp_path)
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    run = lambda: sta.analyze(net, lib, 2.0, pl, lef["macros"])
    net.io = sdc.budget(net, 0.6, "BUF_X4", 4.0)
    base = run()
    io = """set p 2.0
create_clock -name core -period $p [get_ports clk]
set_input_delay [expr 0.3 * $p] -clock core [remove_from_collection [all_inputs] [get_ports clk]]
set_driving_cell -lib_cell BUF_X4 [all_inputs]
set_output_delay 0.6 [all_outputs]   ;# same budget, spelled as SDC
set_load 4.0 [all_outputs]
"""
    net.io, period = sdc.read(io, net, lib)
    t = run()
    assert period == 2.0 and t["slack_by_endpoint"] == base["slack_by_endpoint"] and t["whs_ns"] == base["whs_ns"]

    r = sdc._Reader(net, lib)
    acc = {c for _, c in r.c_get_cells(["*.pe.acc*"])}
    assert len(acc) == 4 * 24                                  # every accumulator bit, by its RTL name
    ep = lambda c: f"{net.cells[c][0]}/{lib[net.cells[c][1]].ff['next']}"
    assert any(ep(c) == base["critical_endpoint"] for c in acc)

    net.io, _ = sdc.read(io + "set_multicycle_path 2 -setup -to [get_cells *.pe.acc*]\n", net, lib)
    m = run()                                                  # capture one cycle later, and (SDC) hold too
    assert all(m["slack_by_endpoint"][ep(c)] == pytest.approx(base["slack_by_endpoint"][ep(c)] + 2.0, abs=1e-3) for c in acc)
    assert all(m["hold_by_ff"][c] == pytest.approx(base["hold_by_ff"][c] - 2.0, abs=1e-6) for c in acc)
    t_req = max(2.0 - s / (2 if k in {ep(c) for c in acc} else 1) for k, s in m["slack_by_endpoint"].items())
    assert m["fmax_mhz"] == pytest.approx(1000 / t_req, abs=0.2) and m["fmax_mhz"] > base["fmax_mhz"]

    net.io, _ = sdc.read(io + "set_multicycle_path 2 -setup -to [get_cells *.pe.acc*]\n"
                              "set_multicycle_path 1 -hold -to [get_cells *.pe.acc*]\n", net, lib)
    assert run()["hold_by_ff"] == pytest.approx(base["hold_by_ff"])  # -hold N-1 puts hold back on the launch edge

    net.io, _ = sdc.read(io + "set_false_path -to [get_cells *.pe.acc*]\nset_clock_uncertainty 0.05\n", net, lib)
    f = run()
    assert not any(ep(c) in f["slack_by_endpoint"] for c in acc) and len(f["slack_by_endpoint"]) == len(base["slack_by_endpoint"]) - 96
    assert f["whs_ns"] == pytest.approx(min(s for c, s in base["hold_by_ff"].items() if c not in acc) - 0.05, abs=1e-3)

    for bad in ("set_max_fanout 8 [current_design]", "set_input_delay 0.1 [get_ports nope]",
                "set p [expr __import__('os')]", "set_false_path -through [get_nets in_reg.a_q*] -through [get_nets in_reg.b_q*]",
                "set_input_delay 0.1 -clock nope [get_ports clr]"):
        with pytest.raises(ValueError):
            sdc.read(bad + "\n", net, lib)


def test_path_tags_make_two_sided_and_through_exceptions_exact(lib, tmp_path):
    p = {"N": 2, "REG_IN": 1}
    net, _ = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", p, tmp_path)
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    io = "create_clock -period 2.0 [get_ports clk]\n"
    def run(extra):
        net.io, _ = sdc.read(io + extra + "\n", net, lib)
        return sta.analyze(net, lib, 2.0, pl, lef["macros"])["slack_by_endpoint"]
    r = sdc._Reader(net, lib)
    B = {f"{net.cells[c][0]}/D" for _, c in r.c_get_cells(["row[0].col[0].pe.acc*"])}
    A, B_sdc = "[get_cells in_reg.*]", "[get_cells row[0].col[0].pe.acc*]"
    base, from_a = run(""), run(f"set_false_path -from {A}")
    only_a = run(f"set_false_path -from [remove_from_collection [all_registers] {A}]")
    both = run(f"set_false_path -from {A} -to {B_sdc}")
    assert any(both[e] != base[e] for e in B)                          # the exception bites ...
    assert all(both[e] == (from_a[e] if e in B else base[e]) for e in base)   # ... exactly on A -> B paths
    assert run("set_false_path -through [get_nets in_reg.*]") == from_a   # every path from in_reg crosses its nets
    mcp = run(f"set_multicycle_path 2 -from {A} -to {B_sdc}")         # A -> B gets 2 cycles, the rest of B keeps 1
    assert all(mcp[e] == pytest.approx(min(from_a[e], only_a[e] + 2.0) if e in B else base[e], abs=2e-4) for e in base)


@pytest.mark.parametrize("launch, capture, want", [((2, 0), (3, 0), (1, -1)), ((3, 0), (2, 0), (1, -1)),
                                                   ((2, 0), (2, 0), (2, 0)), ((2, 0), (2, 0.5), (0.5, -1.5))])
def test_clock_relations(launch, capture, want):
    assert sta.relation(launch, capture) == pytest.approx(want)


def test_two_clock_domains(lib, tmp_path):
    v = tmp_path / "two_clk.v"
    v.write_text("""module two_clk(input clk_a, input clk_b, input [7:0] d, output reg [7:0] q);
  reg [7:0] r, s;
  always @(posedge clk_a) r <= d + 8'd1;
  always @(posedge clk_b) begin s <= r * 8'd3; q <= s ^ r; end
endmodule
""")
    net, _ = synth.synthesize([v], "two_clk", {}, tmp_path / "w")
    with pytest.raises(ValueError):                                   # a clock pin left undefined is an error
        sdc.read("create_clock -period 2 [get_ports clk_a]\n", net, lib)
    def run(pa, pb, extra=""):
        net.io, per = sdc.read(f"create_clock -period {pa} [get_ports clk_a]\ncreate_clock -period {pb} [get_ports clk_b]\n"
                               "set_input_delay 0.2 -clock clk_a [get_ports d]\n" + extra + "\n", net, lib)
        return sta.analyze(net, lib, per, hold=True)
    r = sdc._Reader(net, lib)
    ep = lambda pat: {f"{net.cells[c][0]}/D": c for _, c in r.c_get_cells([pat])}
    s_ep, q_ep, r_ep = ep("s*"), ep("q*"), ep("r*")
    two, one = run(2.0, 3.0), run(1.0, 1.0)            # a->b setup relation is 1.0 ns: same as two 1.0 ns clocks
    assert two["clock"]["domains"] == {"clk_a": 2.0, "clk_b": 3.0}
    assert all(two["slack_by_endpoint"][e] == pytest.approx(one["slack_by_endpoint"][e], abs=1e-4) for e in s_ep)
    assert all(two["slack_by_endpoint"][e] == pytest.approx(one["slack_by_endpoint"][e] + 1.0, abs=1e-4) for e in r_ep)
    assert all(two["hold_by_ff"][c] == pytest.approx(one["hold_by_ff"][c] + 1.0, abs=1e-6) for c in s_ep.values())
    grp = run(2.0, 3.0, "set_clock_groups -asynchronous -group [get_clocks clk_a] -group clk_b")
    assert not set(s_ep) & set(grp["slack_by_endpoint"])            # only r -> s crosses: s goes untimed
    assert all(grp["slack_by_endpoint"][e] == pytest.approx(two["slack_by_endpoint"][e]) for e in r_ep)
    assert set(q_ep) <= set(grp["slack_by_endpoint"])                 # q keeps its clk_b -> clk_b path from s


def test_useful_skew_borrows_time_through_the_clock(lib, tmp_path):
    from chipflow import buffer
    p = {"N": 2, "REG_IN": 1}
    net, _ = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", p, tmp_path)
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    net.io = sdc.budget(net, 0.4, "BUF_X4", 4.0)
    buffer.cts(net, lib, lef, pl)
    T = 1.3
    before = sta.analyze(net, lib, T, pl, lef["macros"])
    assert before["wns_ns"] < 0 and before["hold_failing"] == 0
    r = buffer.useful_skew(net, lib, lef, pl, T)
    after = r["sta"]
    assert r["buffers"] > 0 and after["wns_ns"] > before["wns_ns"] + 0.05 and after["hold_failing"] == 0
    assert after["wns_ns"] == sta.analyze(net, lib, T, pl, lef["macros"])["wns_ns"]     # the returned timing is current
    skew = [c for c, (name, _) in enumerate(net.cells) if name.startswith("skew_")]
    assert len(skew) == r["buffers"] and set(skew) <= sta.clock_network(net, lib)[0]   # all on the clock tree
    assert after["clock"]["skew_ns"] > before["clock"]["skew_ns"]                       # skew is now deliberate
    assert designs.systolic_tb(sim.Simulator(net, lib), p, np.random.default_rng(0))["mismatches"] == 0
    snap = (list(net.cells), net.n_nets, pl.x.copy())            # converged: another call keeps nothing it cannot gain
    again = buffer.useful_skew(net, lib, lef, pl, T, rounds=1)
    assert again["rounds_kept"] == 1 or (net.cells == snap[0] and net.n_nets == snap[1] and np.array_equal(pl.x, snap[2]))


def test_setup_repair_keeps_the_better_strategy(lib, tmp_path):
    from chipflow import buffer
    p = {"N": 2, "REG_IN": 1}
    net, _ = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", p, tmp_path)
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    net.io = sdc.budget(net, 0.4, "BUF_X4", 4.0)
    buffer.cts(net, lib, lef, pl)
    T = 1.3
    area = lambda: sum(lib[t].area for _, t in net.cells)
    cost = lambda net_len: (area() / 1000, area())   # any monotone stand-in for power: the rule is what is tested
    r, t = size.close_setup(net, lib, lef, pl, T, cost)
    c = r["candidates"]
    assert set(c) == {"sizing", "useful_skew+sizing"}
    best = max(c, key=lambda k: (min(c[k]["wns_ns"], 0.0), -c[k]["power_mw"], -c[k]["area_um2"]))
    assert r["chosen"] == best
    assert sta.analyze(net, lib, T, pl, lef["macros"], hold=False)["wns_ns"] == t["wns_ns"] == c[best]["wns_placed_ns"]
    assert all(v["judged_on"] == "routed" for v in c.values())
    assert area() == pytest.approx(c[best]["area_um2"], abs=0.1)              # the kept state is the chosen one
    assert ("useful_skew" in r) == (best == "useful_skew+sizing")
    assert designs.systolic_tb(sim.Simulator(net, lib), p, np.random.default_rng(0))["mismatches"] == 0



def test_vector_engine_matches_python_engine(lib, tmp_path):
    from chipflow import buffer
    p = {"N": 2, "REG_IN": 1}
    net, _ = synth.synthesize([designs.DESIGN_DIR / "systolic_array.v"], "systolic_array", p, tmp_path)
    lef = pdk.load_lef()
    pl = place.place(net, lib, lef)
    buffer.cts(net, lib, lef, pl)
    keys = ("wns_ns", "tns_ns", "whs_ns", "ths_ns", "fmax_mhz", "hold_failing", "endpoints", "slew_violations",
            "cap_violations", "slack_by_endpoint", "hold_by_ff", "net_slack", "critical_path", "critical_endpoint",
            "slow_nets", "cap_nets", "clock")
    cases = [None, sdc.budget(net, 0.6, "BUF_X4", 4.0),
             sdc.read("create_clock -period 2 [get_ports clk]\nset_input_delay 0.3 [get_ports {a_west b_north}]\n"
                      "set_output_delay 0.5 [all_outputs]\nset_clock_uncertainty 0.04\n"
                      "set_false_path -from [get_cells in_reg.a_q*] -to [get_cells row[0].col[0].pe.acc*]\n"
                      "set_multicycle_path 2 -to [get_ports acc]\n", net, lib)[0]]
    for io in cases:
        net.io = io
        for hold in (True, False):
            for nl in (None, {n: 50.0 for n in range(0, net.n_nets, 7)}):      # placed and "routed" lengths
                a = sta.analyze(net, lib, 1.6, pl, lef["macros"], nl, hold=hold, engine="py")
                b = sta.analyze(net, lib, 1.6, pl, lef["macros"], nl, hold=hold, engine="vec")
                assert [k for k in keys if a[k] != b[k]] == []                  # bit for bit, every field
    net.io, _ = sdc.read("create_clock -period 2 [get_ports clk]\nset_false_path -through [get_nets in_reg.a_q*]\n", net, lib)
    with pytest.raises(AssertionError):                                       # tags made on the way: Python only
        sta.analyze(net, lib, 1.6, pl, lef["macros"], engine="vec")
    assert sta.analyze(net, lib, 1.6, pl, lef["macros"])["wns_ns"] is not None   # auto falls back
