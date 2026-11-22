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


def _l_paths(a, b):
    (x1, y1), (x2, y2) = a, b
    out = []
    for corner in ((x2, y1), (x1, y2)):
        p = []
        for (ax, ay), (bx, by) in ((a, corner), (corner, b)):
            if ay == by and ax != bx:
                p += [("h", ay, x) for x in range(min(ax, bx), max(ax, bx))]
            elif ax == bx and ay != by:
                p += [("v", y, ax) for y in range(min(ay, by), max(ay, by))]
        out.append(p)
    return out


def _path_cost(path, ch, cv):
    return sum((ch if k == "h" else cv)[y, x] for k, y, x in path)


def _astar(grid, a, b, ch, cv, margin=6):
    x0, x1 = max(0, min(a[0], b[0]) - margin), min(grid.nx - 1, max(a[0], b[0]) + margin)
    y0, y1 = max(0, min(a[1], b[1]) - margin), min(grid.ny - 1, max(a[1], b[1]) + margin)
    dist = {a: 0.0}
    prev = {}
    heap = [(abs(a[0] - b[0]) + abs(a[1] - b[1]), 0.0, a)]
    while heap:
        _, d, u = heapq.heappop(heap)
        if u == b:
            break
        if d > dist.get(u, math.inf):
            continue
        x, y = u
        for v, edge in (((x + 1, y), ("h", y, x)), ((x - 1, y), ("h", y, x - 1)),
                        ((x, y + 1), ("v", y, x)), ((x, y - 1), ("v", y - 1, x))):
            if not (x0 <= v[0] <= x1 and y0 <= v[1] <= y1):
                continue
            nd = d + (ch if edge[0] == "h" else cv)[edge[1], edge[2]]
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, (u, edge)
                heapq.heappush(heap, (nd + abs(v[0] - b[0]) + abs(v[1] - b[1]), nd, v))
    path, u = [], b
    while u != a:
        u, edge = prev[u]
        path.append(edge)
    return path


def route(net, lib, pl, lef, gcell_rows: float = 2.0, iters: int = 30, log=None) -> dict:
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
    ch, cv = grid.cost_h(), grid.cost_v()
    for c in conns:  # initial pattern routing, congestion-aware but costs refreshed lazily
        a, b = _l_paths(c[1], c[2])
        c[3] = a if _path_cost(a, ch, cv) <= _path_cost(b, ch, cv) else b
        grid.commit(c[3])
        for k, y, x in c[3]:
            if k == "h":
                ch[y, x] = (1 + grid.hist_h[y, x]) * (1 + grid.pres * max(0, grid.use_h[y, x] + 1 - grid.cap_h[y, x]))
            else:
                cv[y, x] = (1 + grid.hist_v[y, x]) * (1 + grid.pres * max(0, grid.use_v[y, x] + 1 - grid.cap_v[y, x]))
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
        bad = [c for c in conns if any((oh if k == "h" else ov)[y, x] for k, y, x in c[3])]
        for c in bad:
            grid.commit(c[3], -1)
            c[3] = _astar(grid, c[1], c[2], grid.cost_h(), grid.cost_v()) if len(bad) < 400 else None
            if c[3] is None:  # many victims: refresh costs per batch to keep it fast
                ch, cv = grid.cost_h(), grid.cost_v()
                c[3] = _astar(grid, c[1], c[2], ch, cv)
            grid.commit(c[3])
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
