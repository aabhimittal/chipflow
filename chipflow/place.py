"""Back end, step 1: floorplan, analytical global placement, row legalization.

Global placement follows ePlace: minimize weighted-average (WA) wirelength plus lambda times
an electrostatic density energy. Cell area is a positive charge; the potential comes from a
Poisson solve (FFT on a mirrored grid = Neumann boundary), so crowded regions push cells out
along long-range field lines instead of only into neighboring bins. Legalization is Tetris.
"""
from __future__ import annotations

import bisect
import dataclasses as dc
import math

import numpy as np
import torch

from chipflow import pdk


@dc.dataclass
class Placement:
    core: tuple                 # (x0, y0, x1, y1) um
    die: tuple
    row_h: float
    site_w: float
    x: np.ndarray               # lower-left x per cell, um
    y: np.ndarray
    flip: np.ndarray            # True = FS orientation (odd rows)
    w: np.ndarray
    h: np.ndarray
    io: dict                    # (port, bit) -> (x, y)
    stats: dict

    def pin_xy(self, c: int, pin: str, macros, typ: str):
        dx, dy = macros[typ].pins.get(pin, (self.w[c] / 2, self.h[c] / 2))
        return self.x[c] + dx, self.y[c] + (self.h[c] - dy if self.flip[c] else dy)


def floorplan(net, lef, util: float = 0.6, margin: float = 4.0):
    macros, (sw, rh) = lef["macros"], lef["site"]
    w = np.array([macros[t].width for _, t in net.cells])
    h = np.array([macros[t].height for _, t in net.cells])
    area = float((w * h).sum())
    rows = max(1, math.ceil(math.sqrt(area / util) / rh))
    cw = math.ceil(area / util / (rows * rh) / sw) * sw
    core = (margin, margin, margin + cw, margin + rows * rh)
    die = (0.0, 0.0, core[2] + margin, core[3] + margin)
    ins = [(p, b) for p, (d, nets) in net.ports.items() if d == "input" for b in range(len(nets))]
    outs = [(p, b) for p, (d, nets) in net.ports.items() if d == "output" for b in range(len(nets))]
    io = {}

    def spread(pins, side):
        for k, pb in enumerate(pins):
            t = (k + 0.5) / len(pins)
            io[pb] = {"W": (die[0], die[3] * (1 - t)), "N": (die[2] * t, die[3]),
                      "E": (die[2], die[3] * (1 - t)), "S": (die[2] * t, die[1])}[side]
    h_in, h_out = (len(ins) + 1) // 2, (len(outs) + 1) // 2
    spread(ins[:h_in], "W"); spread(ins[h_in:], "N"); spread(outs[:h_out], "E"); spread(outs[h_out:], "S")
    return core, die, w, h, io, area


def _pin_arrays(net, lib, macros, io, max_degree):
    """Flatten every net's pins into (net index, movable cell or -1, offset/fixed x, y)."""
    drv, sinks = net.drivers_sinks(lib)
    nid, cell, ox, oy, kept = [], [], [], [], []
    k = 0
    for n in range(net.n_nets):
        pins = ([drv[n]] if n in drv else []) + sinks[n]
        if len(pins) < 2 or len(pins) > max_degree or n in net.const:
            continue
        for p in pins:
            if p[0] == "port":
                x, y = io[(p[1], p[2])]
                cell.append(-1)
            else:
                c, pin = p
                x, y = macros[net.cells[c][1]].pins.get(pin, (0.0, 0.0))
                cell.append(c)
            nid.append(k); ox.append(x); oy.append(y)
        kept.append(n)
        k += 1
    return np.array(nid), np.array(cell), np.array(ox), np.array(oy), k, kept


def global_place(net, lib, lef, util=0.6, iters=1500, seed=0, max_degree=64, log=None, lam_growth=1.02,
                 init=None, inflate=None, net_weight=None, timing=None, timing_every=10, timing_beta=1.0):
    """lam_growth: density weight multiplier per iteration. On the 4x4 array, 1.02 gives 31% lower
    HPWL than 1.05 for +1.6 s; 1.01 gains only 3% more at twice the iterations.
    init: (m, 2) starting centers; inflate: per-cell area factor for the density term (routability);
    net_weight: {net id: weight} on the wirelength term (timing-driven placement).
    timing: callable (core, io, w, h) -> f(pos) giving a smooth worst slack to minimize (slack_objective);
    its gradient is refreshed every `timing_every` iterations and added, scaled so the most critical
    cell's timing force is timing_beta times the mean wirelength force on the cells timing touches
    (a total-norm match concentrates the whole wirelength budget on a few hundred cells)."""
    macros = lef["macros"]
    core, die, w, h, io, area = floorplan(net, lef, util)
    nid, cell, ox, oy, n_nets, kept = _pin_arrays(net, lib, macros, io, max_degree)
    nw = torch.tensor([(net_weight or {}).get(n, 1.0) for n in kept], dtype=torch.float64)
    m = len(net.cells)
    x0, y0, x1, y1 = core
    nb = int(min(128, max(16, 2 ** round(math.log2(math.sqrt(m) / 1.5)))))
    bw, bh = (x1 - x0) / nb, (y1 - y0) / nb
    g = torch.Generator().manual_seed(seed)
    cx = (x0 + x1) / 2 + (torch.rand(m, generator=g, dtype=torch.float64) - 0.5) * (x1 - x0) * 0.1
    cy = (y0 + y1) / 2 + (torch.rand(m, generator=g, dtype=torch.float64) - 0.5) * (y1 - y0) * 0.1
    pos = (torch.tensor(init, dtype=torch.float64) if init is not None else torch.stack([cx, cy], 1)).requires_grad_(True)
    wt, ht = torch.tensor(w), torch.tensor(h)
    nid_t, mov = torch.tensor(nid), torch.tensor(cell >= 0)
    cidx = torch.tensor(np.where(cell >= 0, cell, 0))
    offx = torch.tensor(np.where(cell >= 0, ox - w[np.maximum(cell, 0)] / 2, ox))  # offsets from cell center
    offy = torch.tensor(np.where(cell >= 0, oy - h[np.maximum(cell, 0)] / 2, oy))
    areas = wt * ht * (torch.tensor(inflate) if inflate is not None else 1.0)
    area = float(areas.sum())
    target = min(0.95, util + 0.1)
    ky = torch.fft.fftfreq(2 * nb, dtype=torch.float64)[:, None]
    kx = torch.fft.fftfreq(2 * nb, dtype=torch.float64)[None, :]
    denom = (2 * math.pi) ** 2 * (kx ** 2 + ky ** 2)
    denom[0, 0] = 1.0

    def wa(coord, gamma):
        def lse_mean(s):
            mx = torch.full((n_nets,), -1e30, dtype=coord.dtype).scatter_reduce(0, nid_t, s * coord, "amax")
            e = torch.exp((s * coord - mx[nid_t]) / gamma)
            num = torch.zeros(n_nets, dtype=coord.dtype).index_add(0, nid_t, coord * e)
            den = torch.zeros(n_nets, dtype=coord.dtype).index_add(0, nid_t, e)
            return num / den
        return ((lse_mean(1.0) - lse_mean(-1.0)) * nw).sum()

    def density(p):
        fx = ((p[:, 0] - x0) / bw - 0.5).clamp(0, nb - 1 - 1e-9)
        fy = ((p[:, 1] - y0) / bh - 0.5).clamp(0, nb - 1 - 1e-9)
        ix, iy = fx.floor().long().clamp(max=nb - 2), fy.floor().long().clamp(max=nb - 2)
        tx, ty = fx - ix, fy - iy
        rho = torch.zeros(nb * nb, dtype=p.dtype)
        for dx_, dy_, wgt in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
            rho = rho.index_add(0, (iy + dy_) * nb + ix + dx_, areas * wgt)
        return rho.view(nb, nb) / (bw * bh)

    def energy(rho):
        ext = torch.cat([torch.cat([rho, rho.flip(1)], 1), torch.cat([rho.flip(0), rho.flip(0).flip(1)], 1)], 0)
        spec = torch.fft.fft2(ext - ext.mean())
        psi = torch.fft.ifft2(spec / denom).real[:nb, :nb]
        return (rho * psi).sum()

    def pin_xy(p):
        px = torch.where(mov, p[cidx, 0] + offx, offx)
        py = torch.where(mov, p[cidx, 1] + offy, offy)
        return px, py

    opt = torch.optim.Adam([pos], lr=0.3 * bw)
    lam, hist = None, []
    t_obj = timing(core, io, w, h) if timing else None
    t_grad, lam_t = None, None
    for it in range(iters):
        gamma = 4.0 * bw
        px, py = pin_xy(pos)
        wl = wa(px, gamma) + wa(py, gamma)
        rho = density(pos)
        en = energy(rho)
        if lam is None:
            gvec = torch.autograd.grad(wl, pos, retain_graph=True)[0]
            gw = gvec.abs().sum()                 # as before: default placements stay bit-identical
            gwv = gvec.abs().sum(1)
            ge = torch.autograd.grad(en, pos, retain_graph=True)[0].abs().sum()
            lam = float(gw / ge.clamp(min=1e-12)) * 0.05
        if t_obj is not None and it % timing_every == 0:   # slack gradient at the current positions
            p = pos.detach().clone().requires_grad_(True)
            t_grad = torch.autograd.grad(t_obj(p), p)[0]
            if lam_t is None:
                tc = t_grad.abs().sum(1)
                lam_t = timing_beta * float(gwv[tc > 0].mean() / tc.max().clamp(min=1e-30)) if (tc > 0).any() else 0.0
        loss = wl + lam * en
        opt.zero_grad()
        loss.backward()
        if t_grad is not None:
            pos.grad += lam_t * t_grad
        opt.step()
        with torch.no_grad():
            pos[:, 0].clamp_(min=torch.tensor(x0) + wt / 2, max=torch.tensor(x1) - wt / 2)
            pos[:, 1].clamp_(min=torch.tensor(y0) + ht / 2, max=torch.tensor(y1) - ht / 2)
        over = float(((rho.detach() - target).clamp(min=0) * bw * bh).sum() / area)
        lam *= lam_growth
        if it % 50 == 0:
            hist.append((it, round(wl.item(), 1), round(over, 3)))
            if log:
                log(f"  gp iter {it:4d}  WA-WL {wl.item():10.1f} um  overflow {over:.3f}")
        if over < 0.07 and it > 50:
            break
    c = pos.detach().numpy()
    return core, die, w, h, io, c, {"gp_iters": it + 1, "overflow": round(over, 4), "bins": nb, "gp_hist": hist}


def legalize(core, lef, w, h, centers):
    """Gap-filling greedy legalizer in integer site units. Each row keeps its sorted free
    intervals; a cell takes the nearest gap that fits, searching rows outward from its target
    until the row distance alone exceeds the best cost. Unlike Tetris (append-only per row) it
    backfills holes, so a dense cluster near one edge cannot exhaust the rows."""
    sw, rh = lef["site"]
    x0, y0, x1, y1 = core
    rows = int(round((y1 - y0) / rh))
    nsite = int(round((x1 - x0) / sw))
    S = [[0] for _ in range(rows)]        # free interval starts, sorted
    E = [[nsite] for _ in range(rows)]    # matching ends (exclusive)
    ws = np.round(w / sw).astype(int)
    x, y = np.zeros(len(w)), np.zeros(len(w))

    def best_in_row(r, t, cw, bound):
        st, en = S[r], E[r]
        i = bisect.bisect_right(st, t) - 1
        best = None
        for j in list(range(i, -1, -1)) + list(range(i + 1, len(st))):
            if j <= i and en[j] < t - bound:
                if j < i:
                    continue
            if j > i and st[j] > t + bound:
                break
            if en[j] - st[j] < cw:
                continue
            pos = min(max(t, st[j]), en[j] - cw)
            d = abs(pos - t)
            if best is None or d < best[0]:
                best, bound = (d, j, pos), min(bound, d)
        return best

    for c in np.argsort(centers[:, 0]):
        t = int(round((centers[c, 0] - w[c] / 2 - x0) / sw))
        ty = centers[c, 1] - h[c] / 2
        r0 = int(np.clip(round((ty - y0) / rh), 0, rows - 1))
        best = None
        for k in range(rows * 2):
            r = r0 + (k + 1) // 2 * (1 if k % 2 else -1)
            if not 0 <= r < rows:
                if abs(r - r0) > rows:
                    break
                continue
            dy = abs(y0 + r * rh - ty)
            if best is not None and dy >= best[0]:
                if abs(r - r0) * rh > best[0] + rh:
                    break
                continue
            bound = (best[0] - dy) / sw if best else nsite
            got = best_in_row(r, t, ws[c], bound)
            if got:
                cost = got[0] * sw + dy
                if best is None or cost < best[0]:
                    best = (cost, r, got[1], got[2])
        if best is None:
            raise RuntimeError("legalization failed: cells do not fit in the core rows; lower --util")
        _, r, j, pos = best
        a, b = S[r][j], E[r][j]
        del S[r][j], E[r][j]
        for lo, hi in ((pos + ws[c], b), (a, pos)):   # insert right piece then left piece at index j
            if hi > lo:
                S[r].insert(j, lo)
                E[r].insert(j, hi)
        x[c], y[c] = x0 + pos * sw, y0 + r * rh
    flip = (np.round((y - y0) / rh).astype(int) % 2) == 1
    return x, y, flip


def hpwl(net, lib, pl: Placement, macros, max_degree=None) -> float:
    drv, sinks = net.drivers_sinks(lib)
    total = 0.0
    for n in range(net.n_nets):
        pins = ([drv[n]] if n in drv else []) + sinks[n]
        if len(pins) < 2 or n in net.const or (max_degree and len(pins) > max_degree):
            continue
        xs, ys = [], []
        for p in pins:
            px, py = pl.io[(p[1], p[2])] if p[0] == "port" else pl.pin_xy(p[0], p[1], macros, net.cells[p[0]][1])
            xs.append(px); ys.append(py)
        total += max(xs) - min(xs) + max(ys) - min(ys)
    return total


def place(net, lib, lef, util=0.6, seed=0, log=None, init=None, inflate=None, net_weight=None, timing=None,
          timing_beta=1.0) -> Placement:
    core, die, w, h, io, centers, st = global_place(net, lib, lef, util, seed=seed, log=log, init=init,
                                                    inflate=inflate, net_weight=net_weight, timing=timing,
                                                    timing_beta=timing_beta)
    x, y, flip = legalize(core, lef, w, h, centers)
    pl = Placement(core, die, lef["site"][1], lef["site"][0], x, y, flip, w, h, io, st)
    gx, gy = centers[:, 0] - w / 2, centers[:, 1] - h / 2
    st["legal_disp_avg_um"] = round(float(np.mean(np.abs(x - gx) + np.abs(y - gy))), 3)
    st["hpwl_um"] = round(hpwl(net, lib, pl, lef["macros"], max_degree=64), 1)
    st["core_um"] = [round(core[2] - core[0], 2), round(core[3] - core[1], 2)]
    st["utilization"] = round(float((w * h).sum()) / ((core[2] - core[0]) * (core[3] - core[1])), 3)
    return pl


def timing_weights(t, period_ns, alpha=4.0, frac=0.15) -> dict:
    """Net weights for timing-driven placement: nets within `frac` of the period from the worst slack
    get 1 + alpha * criticality^2. On both benchmarks alpha=4 adds ~3% fmax over an unweighted
    second pass; alpha=8 adds a little more but grows SIMT wirelength 8% (congestion risk)."""
    wns = t["wns_ns"]
    thr = wns + frac * period_ns
    return {n: 1 + alpha * min(1.0, (thr - s) / (thr - wns)) ** 2 for n, s in t["net_slack"].items() if s < thr}


def slack_objective(net, lib, lef, io, w, h, t_graph, period_ns, frac=0.15, tau=0.005):
    """Differentiable timing for global placement: a smooth worst slack over the near-critical part
    of the levelized timing graph (chipflow.sta_vec, graph=True), as a function of cell centers.

    Wire delays follow pin distances; each gate delay is its NLDM table, linear in load within the
    current table segment, at the current input slew; a net's load is its pin capacitance plus wire
    capacitance from its bounding box. Arrivals take a log-sum-exp max (temperature tau ns) and the
    objective is tau * logsumexp(-slack / tau) over endpoints: the smooth worst slack. Returns
    f(pos) -> scalar torch tensor, or None when the design is outside the model (several timed tags).
    shortcut: input slews, launch times and the clock tree are frozen at the operating point; re-run
    on a new placement for a new operating point."""
    g = t_graph
    A, S, load, cpin = g["A"], g["S"], g["load"], g["cpin"]
    timed = [i for i, tag in enumerate(g["tags"]) if tag[0] not in (None, "~clock")]
    if len(timed) != 1 or not g["ends"]:
        return None
    lane = timed[0]
    ends = [e for e in g["ends"] if e[6] == lane]
    wns = ends[0][0]
    ends = [e for e in ends if e[0] < wns + frac * period_ns]
    macros = lef["macros"]
    # near-critical cone: walk the levels backwards from the chosen endpoints
    want = np.zeros(net.n_nets, bool)
    want[[e[2] for e in ends]] = True
    keep = np.zeros(len(g["nin"]), bool)
    for lo, hi, _, part in reversed(g["levels"]):
        if part:
            continue
        sl = slice(lo, hi)
        k = want[g["nout"][sl]] & np.isfinite(A[lane, g["tin"][sl], g["nin"][sl]])
        keep[sl] = k
        want[g["nin"][sl][k]] = True
    idx = np.flatnonzero(keep)
    nin, nout, tin, tout, cell = (g[k][idx] for k in ("nin", "nout", "tin", "tout", "cell"))
    rel = [g["rel_names"][r] for r in g["rel"][idx]]
    # delay = a + b * load, the exact NLDM line through the current load's table segment
    X, Y, V = g["tables"]
    tid = g["dtab"][idx]
    s_in = S[lane, tin, nin]
    c0 = load[nout]
    j = np.clip((Y[tid] <= c0[:, None]).sum(1) - 1, 0, 5)
    from chipflow.sta_vec import _interp
    y0, y1 = Y[tid, j], Y[tid, j + 1]
    d0, d1 = _interp(X, Y, V, tid, s_in, y0), _interp(X, Y, V, tid, s_in, y1)
    b = (d1 - d0) / (y1 - y0)
    a = d0 - b * y0
    # pins (offset from the cell center, as global placement sees them) for wires and bounding boxes
    def pin_off(c, pin):
        m = macros[net.cells[c][1]].pins.get(pin)
        return (m[0] - w[c] / 2, m[1] - h[c] / 2) if m else (0.0, 0.0)
    def pin_rows(pins):
        cs = np.array([p[0] if p[0] != "port" else -1 for p in pins], int)
        ox = np.array([pin_off(*p)[0] if p[0] != "port" else io[(p[1], p[2])][0] for p in pins])
        oy = np.array([pin_off(*p)[1] if p[0] != "port" else io[(p[1], p[2])][1] for p in pins])
        return torch.tensor(cs), torch.tensor(ox), torch.tensor(oy)
    nets = np.unique(np.concatenate([nout, nin, [e[2] for e in ends]]))
    local = {int(n): i for i, n in enumerate(nets)}
    drv, sinks = g["drv"], g["sinks"]
    bb_net, bb_pins = [], []
    for n in np.unique(nout):                        # loads only matter where a modeled gate drives
        for p in ([drv[n]] if n in drv else []) + sinks[n]:
            bb_net.append(local[int(n)])
            bb_pins.append(p)
    bb_c, bb_x, bb_y = pin_rows(bb_pins)
    bb_net = torch.tensor(bb_net)
    sink_c, sink_x, sink_y = pin_rows([(int(c), r) for c, r in zip(cell, rel)] + [e[4] for e in ends])
    drv_c, drv_x, drv_y = pin_rows([drv[int(n)] for n in nin] + [drv[int(e[2])] for e in ends])
    cap = torch.tensor([lib[net.cells[int(c)][1]].pins[r].cap for c, r in zip(cell, rel)] +
                       [(io_load(net, e[4]) if e[4][0] == "port" else lib[net.cells[e[4][0]][1]].pins[e[4][1]].cap) for e in ends])
    r_um, c_um = pdk.WIRE_R_OHM_UM, pdk.WIRE_C_FF_UM
    has_drv = torch.tensor([int(n) in drv for n in nin] + [int(e[2]) in drv for e in ends])
    a_t, b_t = torch.tensor(a), torch.tensor(b)
    nl, cpin_t = len(nets), torch.tensor(cpin[nets])
    # constants: start arrivals (inputs, launches) of the cone, endpoint required times
    base = torch.tensor(A[lane][:, nets])                     # [2, nl]; cone outputs are recomputed
    lo_nin, lo_nout = torch.tensor([local[int(n)] for n in nin]), torch.tensor([local[int(n)] for n in nout])
    tin_t, tout_t = torch.tensor(tin), torch.tensor(tout)
    e_net = torch.tensor([local[int(e[2])] for e in ends])
    e_t = torch.tensor([("r", "f").index(e[3]) for e in ends])
    req = torch.tensor([g["end_req"][e] for e in ends])       # required time at the endpoint pin
    levels = []
    pos_of = {k: i for i, k in enumerate(idx)}
    for lo, hi, _, part in g["levels"]:
        sel = [pos_of[k] for k in range(lo, hi) if k in pos_of]
        if not part and sel:
            levels.append(torch.tensor(sel))
    ne = len(ends)

    def xy(pos, cs, ox, oy):
        mov = cs >= 0
        cc = cs.clamp(min=0)
        return torch.where(mov, pos[cc, 0] + ox, ox), torch.where(mov, pos[cc, 1] + oy, oy)

    def f(pos):
        bx, by = xy(pos, bb_c, bb_x, bb_y)
        big = torch.full((nl,), -1e30, dtype=pos.dtype)
        span = (big.scatter_reduce(0, bb_net, bx, "amax") + big.scatter_reduce(0, bb_net, -bx, "amax") +
                big.scatter_reduce(0, bb_net, by, "amax") + big.scatter_reduce(0, bb_net, -by, "amax"))
        span = torch.where(span > -1e29, span, torch.zeros_like(span))
        ld = cpin_t + c_um * span
        sx, sy = xy(pos, sink_c, sink_x, sink_y)
        dx, dy = xy(pos, drv_c, drv_x, drv_y)
        d = (sx - dx).abs() + (sy - dy).abs()
        wire = torch.where(has_drv, r_um * d * (c_um * d / 2 + cap) * 1e-6, torch.zeros_like(d))
        wk, we = wire[:len(nin)], wire[len(nin):]
        arr = base.clone()
        for li in levels:
            cand = arr[tin_t[li], lo_nin[li]] + wk[li] + a_t[li] + b_t[li] * ld[lo_nout[li]]
            key = tout_t[li] * nl + lo_nout[li]
            m = torch.full((2 * nl,), -1e30, dtype=pos.dtype).scatter_reduce(0, key, cand.detach(), "amax")
            ssum = torch.zeros(2 * nl, dtype=pos.dtype).index_add(0, key, torch.exp((cand - m[key]) / tau))
            outk = torch.unique(key)
            arr = arr.reshape(-1).index_put((outk,), m[outk] + tau * torch.log(ssum[outk])).reshape(2, nl)
        slack = req - arr[e_t, e_net] - we
        return tau * 4 * torch.logsumexp(-slack / (tau * 4), 0)

    return f


def io_load(net, p):
    """Output load of a port pin under the net's constraints (as chipflow.sta uses it)."""
    from chipflow.sta import OUT_LOAD_FF
    return net.io["output_load"].get(p[1], 0.0) if net.io else OUT_LOAD_FF
