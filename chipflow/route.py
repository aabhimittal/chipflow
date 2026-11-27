"""Back end, step 4: global routing with negotiated congestion (PathFinder).

The die is cut into square gcells. Each gcell boundary has a track capacity from the metal2-6
pitches (horizontal layers feed east-west edges, vertical layers north-south), derated for
vias and pin access. Nets are split into 2-pin connections by a Manhattan MST, routed first
as the cheaper L-shape, then connections crossing overflowed edges are ripped up and rerouted
by A* with PathFinder costs: (1 + history) * (1 + present * overuse). History grows on edges
that stay overflowed, so nets negotiate who detours. Detailed routing is not done.
"""
from __future__ import annotations

import heapq
import math

import numba
import numpy as np

from chipflow import pdk


def _mst(pts):
    """Prim on Manhattan distance; returns index pairs."""
    n = len(pts)
    if n < 2:
        return []
    p = np.array(pts)
    inside = np.zeros(n, bool)
    inside[0] = True
    dist = np.abs(p - p[0]).sum(1).astype(float)
    parent = np.zeros(n, int)
    edges = []
    for _ in range(n - 1):
        d = np.where(inside, np.inf, dist)
        j = int(d.argmin())
        edges.append((int(parent[j]), j))
        inside[j] = True
        nd = np.abs(p - p[j]).sum(1)
        upd = nd < dist
        dist[upd], parent[upd] = nd[upd], j
    return edges


class Grid:
    def __init__(self, die, gcell, lef, derate=0.5):
        self.g = gcell
        self.nx, self.ny = math.ceil(die[2] / gcell), math.ceil(die[3] / gcell)
        lay = lef["layers"]
        hcap = sum(gcell / lay[l]["pitch"] for l in pdk.ROUTING_LAYERS if lay[l]["direction"] == "HORIZONTAL")
        vcap = sum(gcell / lay[l]["pitch"] for l in pdk.ROUTING_LAYERS if lay[l]["direction"] == "VERTICAL")
        self.cap_h = np.full((self.ny, self.nx - 1), math.floor(hcap * derate), float)
        self.cap_v = np.full((self.ny - 1, self.nx), math.floor(vcap * derate), float)
        self.use_h, self.use_v = np.zeros_like(self.cap_h), np.zeros_like(self.cap_v)
        self.hist_h, self.hist_v = np.zeros_like(self.cap_h), np.zeros_like(self.cap_v)
        self.pres = 0.5

    def cost_h(self):
        return (1 + self.hist_h) * (1 + self.pres * np.maximum(0, self.use_h + 1 - self.cap_h))

    def cost_v(self):
        return (1 + self.hist_v) * (1 + self.pres * np.maximum(0, self.use_v + 1 - self.cap_v))

    def commit(self, path, sign=1):
        for kind, y, x in path:
            (self.use_h if kind == "h" else self.use_v)[y, x] += sign


@numba.njit(cache=True)
def _cost(use, cap, hist, pres, y, x):
    return (1 + hist[y, x]) * (1 + pres * max(0.0, use[y, x] + 1 - cap[y, x]))   # Grid.cost_h/cost_v, one edge


@numba.njit(cache=True)
def _l_edges(x1, y1, x2, y2, corner_first_h):
    """Edges (kind 0=h, 1=v, y, x) of the L through (x2, y1) if corner_first_h, else through (x1, y2)."""
    n = abs(x2 - x1) + abs(y2 - y1)
    out = np.empty((n, 3), np.int64)
    k = 0
    cx, cy = (x2, y1) if corner_first_h else (x1, y2)
    for (ax, ay), (bx, by) in (((x1, y1), (cx, cy)), ((cx, cy), (x2, y2))):
        if ay == by and ax != bx:
            for x in range(min(ax, bx), max(ax, bx)):
                out[k, 0], out[k, 1], out[k, 2] = 0, ay, x
                k += 1
        elif ax == bx and ay != by:
            for y in range(min(ay, by), max(ay, by)):
                out[k, 0], out[k, 1], out[k, 2] = 1, y, ax
                k += 1
    return out[:k]


@numba.njit(cache=True)
def _path_cost(e, uh, uv, ch, cv, hh, hv, pres):
    s = 0.0
    for i in range(len(e)):
        s += _cost(uh, ch, hh, pres, e[i, 1], e[i, 2]) if e[i, 0] == 0 else _cost(uv, cv, hv, pres, e[i, 1], e[i, 2])
    return s


@numba.njit(cache=True)
def _commit(e, uh, uv, sign):
    for i in range(len(e)):
        if e[i, 0] == 0:
            uh[e[i, 1], e[i, 2]] += sign
        else:
            uv[e[i, 1], e[i, 2]] += sign


@numba.njit(cache=True)
def _pattern(ends, uh, uv, ch, cv, hh, hv, pres):
    """Initial routing: each connection, in order, takes the cheaper of its two L shapes (ties: the
    one through (x2, y1)) under the congestion the earlier ones left. Returns 1 per connection that
    took the (x2, y1) shape."""
    first = np.zeros(len(ends), np.bool_)
    for i in range(len(ends)):
        x1, y1, x2, y2 = ends[i, 0], ends[i, 1], ends[i, 2], ends[i, 3]
        a, b = _l_edges(x1, y1, x2, y2, True), _l_edges(x1, y1, x2, y2, False)
        first[i] = _path_cost(a, uh, uv, ch, cv, hh, hv, pres) <= _path_cost(b, uh, uv, ch, cv, hh, hv, pres)
        _commit(a if first[i] else b, uh, uv, 1)
    return first


@numba.njit(cache=True)
def _reroute(old, x1, y1, x2, y2, nx, ny, margin, uh, uv, ch, cv, hh, hv, pres):
    """Rip up `old`, find the cheapest path with A* inside the connection's box grown by `margin`
    gcells (costs read live from the usage), commit it and return it (edges from b back to a)."""
    _commit(old, uh, uv, -1)
    x0, xm = max(0, min(x1, x2) - margin), min(nx - 1, max(x1, x2) + margin)
    y0, ym = max(0, min(y1, y2) - margin), min(ny - 1, max(y1, y2) + margin)
    W, H = xm - x0 + 1, ym - y0 + 1
    dist = np.full((H, W), np.inf)
    prev = np.full((H, W, 3), -1, np.int64)          # predecessor x, y and the edge kind
    dist[y1 - y0, x1 - x0] = 0.0
    heap = [(float(abs(x1 - x2) + abs(y1 - y2)), 0.0, x1, y1)]
    while heap:
        _, d, x, y = heapq.heappop(heap)
        if x == x2 and y == y2:
            break
        if d > dist[y - y0, x - x0]:
            continue
        for k in range(4):
            if k == 0:
                vx, vy, kind, ey, ex = x + 1, y, 0, y, x
            elif k == 1:
                vx, vy, kind, ey, ex = x - 1, y, 0, y, x - 1
            elif k == 2:
                vx, vy, kind, ey, ex = x, y + 1, 1, y, x
            else:
                vx, vy, kind, ey, ex = x, y - 1, 1, y - 1, x
            if not (x0 <= vx <= xm and y0 <= vy <= ym):
                continue
            nd = d + (_cost(uh, ch, hh, pres, ey, ex) if kind == 0 else _cost(uv, cv, hv, pres, ey, ex))
            if nd < dist[vy - y0, vx - x0]:
                dist[vy - y0, vx - x0] = nd
                prev[vy - y0, vx - x0, 0], prev[vy - y0, vx - x0, 1], prev[vy - y0, vx - x0, 2] = x, y, kind
                heapq.heappush(heap, (nd + abs(vx - x2) + abs(vy - y2), nd, vx, vy))
    out = np.empty((W * H, 3), np.int64)             # a path visits each box gcell at most once
    n, x, y = 0, x2, y2
    while not (x == x1 and y == y1):
        px, py, kind = prev[y - y0, x - x0, 0], prev[y - y0, x - x0, 1], prev[y - y0, x - x0, 2]
        out[n, 0], out[n, 1], out[n, 2] = kind, min(y, py), min(x, px)
        n += 1
        x, y = px, py
    path = out[:n].copy()
    _commit(path, uh, uv, 1)
    return path


def route(net, lib, pl, lef, gcell_rows: float = 2.0, iters: int = 30, log=None) -> dict:
    """Paths are (k, 3) int arrays of edges (kind 0 = horizontal, 1 = vertical; y; x)."""
    macros = lef["macros"]
    grid = Grid(pl.die, gcell_rows * pl.row_h, lef)
    g = grid.g
    drv, sinks = net.drivers_sinks(lib)
    conns, local_len = [], {}
    for n in range(net.n_nets):
        pins = ([drv[n]] if n in drv else []) + sinks[n]
        if len(pins) < 2 or n in net.const:
            continue
        xy = [pl.io[(p[1], p[2])] if p[0] == "port" else pl.pin_xy(p[0], p[1], macros, net.cells[p[0]][1]) for p in pins]
        cells = sorted({(min(int(x / g), grid.nx - 1), min(int(y / g), grid.ny - 1)) for x, y in xy})
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        local_len[n] = (max(xs) - min(xs) + max(ys) - min(ys)) if len(cells) == 1 else 0.0
        for i, j in _mst(cells):
            conns.append([n, cells[i], cells[j], None])
    ends = np.array([(a[0], a[1], b[0], b[1]) for _, a, b, _ in conns], np.int64).reshape(-1, 4)
    args = (grid.cap_h, grid.cap_v, grid.hist_h, grid.hist_v)
    first = _pattern(ends, grid.use_h, grid.use_v, *args, grid.pres)
    for c, e, f in zip(conns, ends, first):
        c[3] = _l_edges(e[0], e[1], e[2], e[3], bool(f))
    hist = []
    for it in range(iters):
        oh, ov = grid.use_h > grid.cap_h, grid.use_v > grid.cap_v
        overflow = float(np.maximum(0, grid.use_h - grid.cap_h).sum() + np.maximum(0, grid.use_v - grid.cap_v).sum())
        hist.append((it, overflow))
        if log:
            log(f"  route iter {it:2d}  overflow {overflow:.0f}")
        if overflow == 0:
            break
        grid.hist_h += oh
        grid.hist_v += ov
        grid.pres *= 1.6
        flat = np.concatenate([c[3] for c in conns])
        hit = np.where(flat[:, 0] == 0, oh[flat[:, 1], np.minimum(flat[:, 2], grid.nx - 2)],
                       ov[np.minimum(flat[:, 1], grid.ny - 2), flat[:, 2]])
        cut = np.cumsum([0] + [len(c[3]) for c in conns])
        bad = [conns[i] for i in np.flatnonzero(np.add.reduceat(hit, cut[:-1]) > 0) if cut[i + 1] > cut[i]]
        for c in bad:   # rip up and reroute, in connection order, each against the live congestion
            c[3] = _reroute(c[3], c[1][0], c[1][1], c[2][0], c[2][1], grid.nx, grid.ny, 6,
                            grid.use_h, grid.use_v, *args, grid.pres)
    net_len = dict(local_len)
    for n, _, _, path in conns:
        net_len[n] = net_len.get(n, 0.0) + len(path) * g
    over_h, over_v = np.maximum(0, grid.use_h - grid.cap_h), np.maximum(0, grid.use_v - grid.cap_v)
    return {"wirelength_um": round(float(sum(net_len.values())), 1), "connections": len(conns),
            "overflow": int(over_h.sum() + over_v.sum()), "overflow_edges": int((over_h > 0).sum() + (over_v > 0).sum()),
            "gcell_um": round(g, 3), "grid": [grid.nx, grid.ny], "cap_h": int(grid.cap_h.flat[0]),
            "cap_v": int(grid.cap_v.flat[0]), "iterations": len(hist),
            "max_util": round(float(max((grid.use_h / grid.cap_h).max(), (grid.use_v / grid.cap_v).max())), 3),
            "net_len": net_len, "grid_obj": grid, "conns": conns}


def inflation(pl, grid, hot: float = 0.85, gain: float = 3.0, cap: float = 2.0) -> np.ndarray:
    """Routability-driven placement: per-cell area factor for the density term, growing with the
    routing utilization of the cell's gcell above `hot`, so re-placement spreads congested regions."""
    from chipflow.layout import congestion_map
    u = congestion_map(grid)
    gx = np.clip(((pl.x + pl.w / 2) / grid.g).astype(int), 0, grid.nx - 1)
    gy = np.clip(((pl.y + pl.h / 2) / grid.g).astype(int), 0, grid.ny - 1)
    return np.clip(1 + gain * np.maximum(0, u[gy, gx] - hot), 1, cap)
