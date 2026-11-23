# chipflow

An open digital EDA flow, written to be read and extended: Verilog goes in, and a
verified, placed, timed and globally routed layout on the Nangate45 open cell library comes
out. The two built-in benchmarks are the building blocks of AI accelerators: a TPU-style
systolic array and a GPU-style SIMT core slice.

```
Verilog ─► lint + synthesis ─► gate-level simulation ─► floorplan ─► global placement ─► legalization
           (Yosys → Nangate45)  vs golden model           (ePlace-style, PyTorch)   (gap-filling)
       ─► clock tree + buffering ─► sizing + hold fix ─► global routing ─► post-route timing + power ─► DEF, SVG, report.json
          (CTS, design-rule repair)  (propagated-clock STA) (PathFinder A*)   (re-verified netlist activity)
```

| Stage | Front end / back end | What it does | Code |
|---|---|---|---|
| Synthesis | front | Yosys (YoWASP build, pip-installable) maps RTL onto Nangate45 cells; lint warnings are reported | `synth.py` |
| Verification | front | Bit-parallel gate-level simulator: 64 independent test vectors per pass, cell behavior compiled from the Liberty `function` strings, compared with a Python golden model | `sim.py`, `designs.py` |
| Floorplan + placement | back | Core sized for a target utilization; global placement minimizes weighted-average wirelength plus an electrostatic density energy (FFT Poisson solve); a gap-filling legalizer snaps cells to rows and sites. **Timing-driven**: near-critical nets are weighted and the design re-placed. **Routability-driven**: cells in congested gcells are inflated and re-placed until routing overflow clears | `place.py` |
| Static timing | back | Rise/fall propagation with unateness through Liberty NLDM delay and slew tables; Elmore wire delay from placed or routed lengths; propagated clock through the clock tree (latency and skew per flip-flop); setup on latest arrivals, hold on earliest (register to register); slew and capacitance limit checks; backward required-time pass for slack on every net; critical path trace | `sta.py` |
| Clock tree + buffers | back | Clock-tree synthesis: each clock's flip-flops are split by recursive median bisection into groups of ≤ 16 under a `CLKBUF_X3`, level by level up to the clock pin. Design-rule repair: nets driving > 16 pins get a `BUF_X4` tree, then drivers still over the slew or capacitance limit are upsized. Hold fixing: `BUF_X1` delay cells in front of flip-flops whose data arrives too early, where setup slack allows | `buffer.py` |
| Timing repair | back | Gate sizing: worst-slack cells move up the library's drive strengths (X1 → X2 → X4…) in batches kept only if WNS improves; runs after buffering and again on routed lengths | `size.py` |
| Global routing | back | Gcell grid with track capacities from metal2–6 pitches; MST net decomposition; L-shape initial routes; negotiated rip-up and reroute (PathFinder) with A* | `route.py` |
| Power | back | Switching power from simulated toggle rates on routed net capacitance, clock-tree power (every clock net switches twice a cycle), Liberty leakage | `power.py` |
| Outputs | back | DEF (opens in KLayout or OpenROAD), layout and congestion SVGs, JSON report | `layout.py` |

## Demo

[**chipflow layout explorer**](https://aabhimittal.github.io/chipflow/) is a static page in `demo/`
built from a real run of both designs. It opens with a short guide (how to use it, how to read it,
things to try), then:
- **Layout:** the placed chip. Hover any cell to see its block, cell type and critical-path stage;
  color by block or by flip-flop vs. logic; overlay the critical path, routing congestion and I/O pins.
- **Critical path:** a gate-by-gate table that highlights each gate on the layout.
- **Slack histogram:** the timing margin at every endpoint.
- **Flow stages and source:** what each of the 7 stages produced, and the Verilog source.

```bash
python demo/build_data.py                  # rerun both designs, write demo/data.json (~35 s)
python -m http.server -d demo 8000         # open http://localhost:8000
```

`.github/workflows/pages.yml` deploys it on pushes to `main` that touch `demo/`. Enable it once
under *Settings → Pages → Source: GitHub Actions*.

## Quickstart

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[test]"
python -m chipflow fetch-pdk          # Nangate45 from OpenROAD-flow-scripts, checksum-pinned
pytest -q                            # ~15 s

python -m chipflow flow --design systolic --param N=4 --clock-ns 2.0 --out runs/sa4
python -m chipflow flow --design simt --clock-ns 3.0 --out runs/simt
python -m chipflow flow --verilog my.v --top my_top --clock-ns 2.0 --out runs/mine   # your own RTL
```

## Results (CPU, 4 cores, defaults)

| | 4×4 systolic array (INT8 MAC, 24-bit acc) | SIMT slice (4 lanes, 2 warps, 16-bit) |
|---|---|---|
| Cells / flip-flops | 12,736 / 640 | 11,094 / 1,088 |
| Cell area | 16,573 µm² | 17,181 µm² |
| Gate-level verification | 1,024 outputs, 0 mismatches | 50,944 lane outputs, 0 mismatches |
| Core (60% utilization) | 166 × 167 µm | 169 × 169 µm |
| Clock tree | 68 `CLKBUF_X3`, latency 132–148 ps, **skew 16 ps** | 136 `CLKBUF_X3`, latency 104–132 ps, **skew 27 ps** |
| Design-rule repair | 92 nets buffered (184 `BUF_X4`); slew/cap violations 16/64 → **0/0** | 223 nets buffered (553 `BUF_X4`), 75 drivers upsized; 121/229 → **0/0** |
| Routed wirelength / overflow | 129 mm / 0 | 182 mm / 0 (after 1 routability round) |
| Routed fmax at 2.0 / 3.0 ns | **651 MHz** (591 before buffering, 564 before M1) | **587 MHz** (440 before buffering, 425 before M1) |
| Hold, worst slack | +0.134 ns, no delay buffers needed | +0.115 ns, none needed |
| Tightest clock met after routing | **1.4 ns (716 MHz)**: 1,002 cells upsized | **1.8 ns (585 MHz)** |
| Critical path | 29 stages, register → signed multiplier → adder → accumulator | 30 stages, instruction → register-file read → ALU |
| Power at the target clock | 4.35 mW (clock tree 0.72 mW) | 4.87 mW (clock tree 0.78 mW) |
| Runtime, whole flow | 18 s | 51 s |

Timing-driven placement was checked against a control: an unweighted second placement pass
alone gains +1.2% (systolic) and +0.4% (SIMT) fmax; the net weights add about +3.5% and +2.9% on top.

<p>
<img src="docs/systolic_4x4_layout.png" width="49%" alt="Placed 4x4 systolic array: 16 processing-element tiles in distinct colors, critical path in red">
<img src="docs/simt_4lane_layout.png" width="49%" alt="Placed SIMT core: four lane regions, register-file flip-flops darker, critical path in red">
</p>

Left: the 16 processing elements of the systolic array land as 16 tiles. Right: the SIMT
slice splits into its 4 lanes, with each lane's register-file flip-flops (darker) inside it.
Red is the post-route critical path.

## Things the flow already caught

- **A signed-arithmetic RTL bug.** The first systolic PE wrote
  `acc <= clr ? {ACC{1'b0}} : acc + a_in * b_in`. In Verilog, the unsigned concatenation makes
  the whole `?:` expression unsigned, which silently zero-extends the signed operands: −1 × 5
  came out as 1275. Gate-level simulation against the golden model failed on 234 of 256
  outputs; the fix was an `if/else`.
- **A legalizer that could not backfill.** Tetris-style append-only rows failed on the SIMT
  core at 60% utilization, because a dense cluster near one edge exhausted the rows to its
  right. The legalizer now keeps each row's free intervals and fills the nearest gap.
- **Buffering mattered more than sizing.** The systolic array's 8-bit operands fan out to 31
  partial-product gates each from a single `AND2_X1`, and SIMT's register-file decoders drive up
  to 66 pins. Buffering those nets (and upsizing weak drivers) cleared every slew and capacitance
  violation and raised routed fmax 10% (systolic) and 33% (SIMT), more than sizing alone did.
- **Routing that could not converge.** SIMT kept 10 overflowed edges however long the router
  negotiated: 15 connections needed an edge with 14 tracks and every detour was full, so the
  congestion just moved around. The fix belonged in placement: inflate cells in hot gcells and
  re-place, which cleared it in one round.
- **A placement schedule that quit too early.** Growing the density weight 5% per iteration
  spread cells before wirelength could organize them. 2% per iteration cut systolic HPWL by
  31% and more than doubled the post-placement slack.

## Limits

- **Not signoff.** Nangate45 is purposely non-manufacturable; there is no DRC, LVS or
  parasitic extraction, and timing uses one typical corner.
- **Global routing only.** No track assignment or detailed routing. Overflow is now cleared by
  routability-driven re-placement (SIMT needed one round), at a cost of a few percent fmax.
- **Clock tree is geometric, not skew-balanced.** Median bisection keeps skew at 16–27 ps here,
  but there is no wire snaking or buffer sizing to drive skew to zero, and one clock domain only.
- **Hold is register to register.** Input ports have no external timing (no input delays), so
  paths from them are not hold-checked; neither design needed delay buffers.
- **Input ports are ideal drivers.** Port nets such as SIMT's instruction bits (fanout ~600) are not
  buffered, because no driving cell is modeled for them.
- **Power is a lower bound.** Cell internal power is not modeled. Nangate45 does not declare the
  unit of its `internal_power` tables, and the two Liberty readings (V²·fF = fJ, or V·mA·ns = pJ)
  differ 1000×; neither matches a 45 nm gate cleanly, so this waits for a calibration against a
  reference tool.
- **Single-threaded Python** for the router, STA and legalizer; placement uses PyTorch on CPU.

## Roadmap

1. Input and output constraints (input delays, driving cells, output loads), so port paths are
   timed and buffered like internal ones.
2. Skew-balanced clock trees (wire snaking, buffer sizing) and multiple clock domains.
3. Differentiable STA: slack gradients straight into placement, replacing net weights.
4. Track assignment, then detailed routing.
5. GPU kernels for placement density, STA propagation and routing cost updates, for designs
   above 100k cells.

## License

Apache-2.0. Nangate45 is downloaded at setup from OpenROAD-flow-scripts under its own
Apache-2.0 terms and is not redistributed here.
