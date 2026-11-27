"""Back end: track assignment, then detailed routing on the metal tracks.

The routing grid has a column every 0.19 um (metal2 pitch) and a row every 0.14 um (metal3 pitch).
Layers route in their preferred direction only:

  metal2  vertical    every column, every row        (width 0.07, spacing 0.07)
  metal3  horizontal  every row, every column        (width 0.07, spacing 0.07)
  metal4  vertical    every 2nd column, every 2nd row (width 0.14, spacing 0.14: 0.38 / 0.28 um apart)
  metal5  horizontal  every 2nd row, every 2nd column
  metal6  vertical    every 2nd column, every 2nd row

so neighbouring tracks and neighbouring wire ends of different nets keep the layer's spacing as
long as no grid node is shared. A via joins two adjacent layers where both have a node. Each cell
pin is reached by a via1 from its metal1 shape to the nearest free metal2 node.

1. Track assignment: every straight run of a net's global route that spans 2+ gcell boundaries
   gets a layer and a track in its gcell panel (left-edge interval packing, long runs on the upper
   layer). Detailed routing then prefers those tracks.
2. Detailed routing: each net grows a tree from its first pin, adding the other pins by Dijkstra
   on the track graph inside its global route (gcells, dilated by one) with negotiated congestion
   (PathFinder): nodes used by several nets get more expensive each round until none is shared.

Checked: shorts (shared nodes), opens (every pin of a net in one connected piece), and that every
wire lies on its layer's tracks. Not modeled: via-to-via cut spacing, minimum area, end-of-line
rules, and pin shapes beyond their center point (pin access snaps to the nearest metal2 node).
"""
from __future__ import annotations

import heapq

import numba
import numpy as np

LAYERS = ("metal2", "metal3", "metal4", "metal5", "metal6")
VERT = np.array([1, 0, 1, 0, 1], np.int64)
STRIDE = np.array([1, 1, 2, 2, 2], np.int64)
VIAS = ("via2_8", "via3_2", "via4_0", "via5_0")   # between LAYERS[l] and LAYERS[l + 1]
PIN_VIA = "via1_4"
DX, DY, VIA_COST = 0.19, 0.14, 0.5               # um; a via costs like 0.5 um of wire


@numba.njit(cache=True)
def _node(l, r, c, R, C):
    return (l * R + r) * C + c


@numba.njit(cache=True)
def _route_net(pins, allowed, colg, rowg, pref, owner, net_id, occ, hist, pres, R, C, stamp, mark, seen, tmark, dist, prev):
    """Grow a tree over `pins` (node ids, first is the root). Returns (ok, nodes, edges).
    mark / seen / tmark are per-node stamps (assigned track, visited in this search, in the tree),
    so nothing is cleared between searches."""
    stamp[0] += 1
    s_net = stamp[0]
    for n in pref:
        mark[n] = s_net                              # this net's assigned tracks
    tree = [pins[0]]
    tmark[pins[0]] = s_net
    edges = []
    r0, c0 = (pins[0] // C) % R, pins[0] % C
    bb = np.array([r0, r0, c0, c0])                  # the tree's bounding box (rows, columns), for A*
    for k in range(1, len(pins)):
        if tmark[pins[k]] == s_net:
            continue
        stamp[1] += 1
        s = stamp[1]
        src = pins[k]
        dist[src] = 0.0
        prev[src] = -1
        seen[src] = s
        heap = [(0.0, 0.0, src)]
        goal = -1
        while heap:
            _, d, u = heapq.heappop(heap)
            if tmark[u] == s_net:
                goal = u
                break
            if d > dist[u]:
                continue
            c = u % C
            r = (u // C) % R
            l = u // (R * C)
            st = STRIDE[l]
            for j in range(4):
                if j < 2:                            # along the track
                    step = st if j == 0 else -st
                    if VERT[l] == 1:
                        r2, c2, l2, base = r + step, c, l, DY * st
                    else:
                        r2, c2, l2, base = r, c + step, l, DX * st
                else:                                # via up or down
                    l2 = l + 1 if j == 2 else l - 1
                    r2, c2, base = r, c, VIA_COST
                    if l2 < 0 or l2 >= len(STRIDE) or r % STRIDE[l2] != 0 or c % STRIDE[l2] != 0:
                        continue
                if r2 < 0 or r2 >= R or c2 < 0 or c2 >= C:
                    continue
                if not allowed[rowg[r2], colg[c2]]:
                    continue
                v = _node(l2, r2, c2, R, C)
                if owner[v] >= 0 and owner[v] != net_id:
                    continue
                w = base * (1.0 + hist[v]) * (1.0 + pres * occ[v])
                if mark[v] == s_net:
                    w *= 0.5
                nd = d + w
                if seen[v] != s or nd < dist[v]:
                    seen[v] = s
                    dist[v] = nd
                    prev[v] = u
                    # lower bound to the tree: distance to its box, at the 0.5 assigned-track discount
                    h = 0.5 * (DX * max(0, bb[2] - c2, c2 - bb[3]) + DY * max(0, bb[0] - r2, r2 - bb[1]))
                    heapq.heappush(heap, (nd + h, nd, v))
        if goal < 0:
            return False, np.zeros(0, np.int64), np.zeros((0, 2), np.int64)
        v = goal
        while v != src:
            u = prev[v]
            edges.append((u, v))
            if tmark[u] != s_net:
                tmark[u] = s_net
                tree.append(u)
                ru, cu = (u // C) % R, u % C
                bb[0], bb[1], bb[2], bb[3] = min(bb[0], ru), max(bb[1], ru), min(bb[2], cu), max(bb[3], cu)
            v = u
        if tmark[src] != s_net:
            tmark[src] = s_net
            tree.append(src)
            rs, cs = (src // C) % R, src % C
            bb[0], bb[1], bb[2], bb[3] = min(bb[0], rs), max(bb[1], rs), min(bb[2], cs), max(bb[3], cs)
    nodes = np.array(tree, np.int64)
    e = np.empty((len(edges), 2), np.int64)
    for i in range(len(edges)):
        e[i, 0], e[i, 1] = edges[i]
    return True, nodes, e


def _runs(path, kind):
    """Straight runs of one kind in a global route path: [(line, lo, hi)] in gcell edge units."""
    e = path[path[:, 0] == kind]
    out = []
    if len(e):
        line, pos = (e[:, 1], e[:, 2]) if kind == 0 else (e[:, 2], e[:, 1])
        o = np.lexsort((pos, line))
        line, pos = line[o], pos[o]
        start = 0
        for i in range(1, len(pos) + 1):
            if i == len(pos) or line[i] != line[start] or pos[i] != pos[i - 1] + 1:
                out.append((int(line[start]), int(pos[start]), int(pos[i - 1])))
                start = i
    return out


def assign_tracks(conns, g, R, C):
    """Track assignment: {net: [(layer, track line, lo, hi)]} with lo..hi in grid units along the
    track, plus counts. Runs of a panel go to the first free track (left edge), upper layer first
    for runs of 4+ gcells."""
    runs = {0: {}, 1: {}}                            # kind -> panel -> [(lo, hi, net)]
    for n, _, _, path in conns:
        for kind in (0, 1):
            for line, lo, hi in _runs(path, kind):
                if hi - lo >= 1:
                    runs[kind].setdefault(line, []).append((lo, hi, n))
    out, assigned, total = {}, 0, 0
    for kind, panels in runs.items():
        lay = (1, 3) if kind == 0 else (2, 4)        # horizontal runs: metal3/metal5; vertical: metal4/metal6
        for line, items in panels.items():
            free = {}                                # (layer, track) -> last occupied gcell
            for l in lay:
                st = int(STRIDE[l])
                span = (line * g / DY, (line + 1) * g / DY) if kind == 0 else (line * g / DX, (line + 1) * g / DX)
                for t in range(int(np.ceil(span[0])), int(np.floor(span[1])) + 1):
                    if t % st == 0 and t < (R if kind == 0 else C):
                        free[(l, t)] = -2
            for lo, hi, n in sorted(items):
                total += 1
                order = sorted(free, key=lambda k: (k[0] != (lay[1] if hi - lo >= 3 else lay[0]), k))
                for k in order:
                    if free[k] < lo - 1:              # one gcell of gap to the previous run on the track
                        free[k] = hi + 1
                        unit = g / (DX if kind == 0 else DY)
                        out.setdefault(n, []).append((k[0], k[1], int(lo * unit), int((hi + 2) * unit)))
                        assigned += 1
                        break
    return out, {"runs": total, "assigned": assigned}


def route(net, lib, pl, lef, rt, iters=40, log=None, track_assign=True) -> dict:
    """Detailed routing after global routing `rt` (chipflow.route). Returns stats and the wires."""
    macros = lef["macros"]
    W, H = pl.die[2], pl.die[3]
    C, R, L = int(W / DX), int(H / DY), len(LAYERS)
    N = L * R * C
    g = rt["gcell_um"]
    gx_n, gy_n = rt["grid"]
    colg = np.minimum(((np.arange(C) + 0.5) * DX / g).astype(np.int64), gx_n - 1)
    rowg = np.minimum(((np.arange(R) + 0.5) * DY / g).astype(np.int64), gy_n - 1)
    drv, sinks = net.drivers_sinks(lib)
    # pin access: nearest free metal2 node (other nets' pins are reserved)
    owner = np.full(N, -1, np.int64)
    pins = {}
    moved = 0
    for n in range(net.n_nets):
        ps = ([drv[n]] if n in drv else []) + sinks[n]
        if len(ps) < 2 or n in net.const:
            continue
        nodes = []
        for p in ps:
            x, y = pl.io[(p[1], p[2])] if p[0] == "port" else pl.pin_xy(p[0], p[1], macros, net.cells[p[0]][1])
            c0, r0 = min(C - 1, max(0, int(round(x / DX - 0.5)))), min(R - 1, max(0, int(round(y / DY - 0.5))))
            best = None
            for rad in range(0, 6):
                for dr in range(-rad, rad + 1):
                    for dc in range(-rad, rad + 1):
                        if max(abs(dr), abs(dc)) != rad:
                            continue
                        r, c = r0 + dr, c0 + dc
                        if 0 <= r < R and 0 <= c < C:
                            v = _node(0, r, c, R, C)
                            if owner[v] in (-1, n) and (best is None or (dc * dc + dr * dr) < best[0]):
                                best = (dc * dc + dr * dr, v)
                if best:
                    break
            owner[best[1]] = n
            moved += best[0] > 0
            nodes.append(best[1])
        pins[n] = np.array(list(dict.fromkeys(nodes)), np.int64)
    # guides: the global route's gcells per net, dilated by one
    guide = {}
    for n, a, b, path in rt["conns"]:
        s = guide.setdefault(n, set())
        s.update([a, b])
        for k, y, x in path.tolist():
            s.update([(x, y), (x + 1, y)] if k == 0 else [(x, y), (x, y + 1)])
    ta, ta_stats = assign_tracks(rt["conns"], g, R, C) if track_assign else ({}, {"runs": 0, "assigned": 0})
    pref = {}
    for n, items in ta.items():
        v = []
        for l, t, lo, hi in items:
            st = int(STRIDE[l])
            for p in range(max(0, lo - lo % st), hi + 1, st):
                r, c = (t, p) if not VERT[l] else (p, t)
                if 0 <= r < R and 0 <= c < C:
                    v.append(_node(l, r, c, R, C))
        pref[n] = np.array(v, np.int64)

    def mask(n, grow):
        m = np.zeros((gy_n, gx_n), bool)
        if grow < 0:
            m[:] = True
            return m
        for x, y in guide.get(n, ()):
            m[max(0, y - grow):y + grow + 1, max(0, x - grow):x + grow + 1] = True
        for v in pins[n]:
            m[rowg[v // C % R], colg[v % C]] = True
        return m

    occ, hist = np.zeros(N, np.float64), np.zeros(N, np.float64)
    stamp, mark, seen, tmark = np.zeros(2, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64)
    dist, prev = np.zeros(N), np.zeros(N, np.int64)
    sol, grow = {}, {n: 1 for n in pins}
    nets = sorted(pins, key=lambda n: len(guide.get(n, ())))
    todo, pres, hist_log = list(nets), 0.5, []
    empty = np.zeros(0, np.int64)
    for it in range(iters):
        for n in todo:
            if n in sol:
                occ[sol[n][0]] -= 1
            ok = False
            while not ok:
                ok, nodes, edges = _route_net(pins[n], mask(n, grow[n]), colg, rowg, pref.get(n, empty), owner, n,
                                              occ, hist, pres, R, C, stamp, mark, seen, tmark, dist, prev)
                if not ok:
                    if grow[n] == -1:
                        raise RuntimeError(f"net {n}: pins not connectable even over the whole die")
                    grow[n] = -1 if grow[n] >= 3 else grow[n] + 1
            occ[nodes] += 1
            sol[n] = (nodes, edges)
        over = occ > 1
        shorts = int(over.sum())
        hist_log.append((it, len(todo), shorts))
        if log:
            log(f"  detail iter {it:2d}  nets routed {len(todo):5d}  shared nodes {shorts}")
        if not shorts:
            break
        hist[over] += 1.0
        pres *= 1.5
        todo = [n for n in nets if over[sol[n][0]].any()]
    return {"sol": sol, "pins": pins, "grid": (L, R, C), "track_assignment": ta_stats,
            "pin_access_moved": moved, "iterations": hist_log, **check(sol, pins, (L, R, C))}


def check(sol, pins, grid) -> dict:
    """Shorts, opens and off-track wires, plus wirelength per layer and via counts."""
    L, R, C = grid
    used = np.zeros(L * R * C, np.int64)
    opens = off = 0
    wl = np.zeros(L)
    vias = np.zeros(L - 1, np.int64)
    for n, (nodes, edges) in sol.items():
        used[np.unique(nodes)] += 1
        parent = {int(v): int(v) for v in nodes}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for u, v in edges.tolist():
            lu, lv = u // (R * C), v // (R * C)
            ru, rv, cu, cv = u // C % R, v // C % R, u % C, v % C
            st = int(STRIDE[lu])
            if lu == lv:
                ok = (ru % st == 0 and cu % st == 0 and ((VERT[lu] and cu == cv and abs(ru - rv) == st) or
                                                        (not VERT[lu] and ru == rv and abs(cu - cv) == st)))
                wl[lu] += DY * st if VERT[lu] else DX * st
            else:
                ok = abs(lu - lv) == 1 and ru == rv and cu == cv and all(ru % int(STRIDE[k]) == 0 and cu % int(STRIDE[k]) == 0 for k in (lu, lv))
                vias[min(lu, lv)] += 1
            off += not ok
            parent[find(u)] = find(v)
        opens += len({find(int(p)) for p in pins[n]}) > 1
    return {"shorts": int((used > 1).sum()), "opens": opens, "off_track": off, "nets": len(sol),
            "wirelength_um": {LAYERS[l]: round(float(wl[l]), 1) for l in range(L)},
            "vias": {VIAS[l]: int(vias[l]) for l in range(L - 1)} | {PIN_VIA: int(sum(len(p) for p in pins.values()))}}


def def_wires(det) -> dict:
    """{net: [DEF ROUTED pieces]}: straight wires merged per track, vias, and via1 at pin access."""
    L, R, C = det["grid"]
    d = lambda v: int(round(v * 2000))
    xy = lambda v: (d((v % C + 0.5) * DX), d((v // C % R + 0.5) * DY))
    out = {}
    for n, (nodes, edges) in det["sol"].items():
        segs, pieces = {}, []
        for u, v in edges.tolist():
            lu, lv = u // (R * C), v // (R * C)
            if lu != lv:
                x, y = xy(u)
                pieces.append(f"{LAYERS[min(lu, lv)]} ( {x} {y} ) {VIAS[min(lu, lv)]}")
            else:
                key = (lu, u % C) if VERT[lu] else (lu, u // C % R)
                a, b = (u // C % R, v // C % R) if VERT[lu] else (u % C, v % C)
                segs.setdefault(key, []).append((min(a, b), max(a, b)))
        for (l, line), iv in segs.items():
            iv.sort()
            lo, hi = iv[0]
            for a, b in iv[1:] + [(None, None)]:
                if a is not None and a <= hi:
                    hi = max(hi, b)
                    continue
                if VERT[l]:
                    p1, p2 = (d((line + 0.5) * DX), d((lo + 0.5) * DY)), (d((line + 0.5) * DX), d((hi + 0.5) * DY))
                else:
                    p1, p2 = (d((lo + 0.5) * DX), d((line + 0.5) * DY)), (d((hi + 0.5) * DX), d((line + 0.5) * DY))
                pieces.append(f"{LAYERS[l]} ( {p1[0]} {p1[1]} ) ( {p2[0]} {p2[1]} )")
                if a is not None:
                    lo, hi = a, b
        for v in det["pins"][n]:
            x, y = xy(int(v))
            pieces.append(f"metal1 ( {x} {y} ) {PIN_VIA}")
        out[n] = pieces
    return out
