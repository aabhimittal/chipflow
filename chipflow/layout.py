"""Outputs: DEF (placed design, opens in KLayout/OpenROAD) and SVG pictures of layout and congestion."""
from __future__ import annotations

import colorsys
import re
from html import escape

import numpy as np

DBU = 2000  # Nangate45 LEF database units per micron


def write_def(path, net, pl, lef):
    d = lambda v: int(round(v * DBU))
    sw, rh = lef["site"]
    site = "FreePDK45_38x28_10R_NP_162NW_34O"
    rows = int(round((pl.core[3] - pl.core[1]) / rh))
    nsite = int(round((pl.core[2] - pl.core[0]) / sw))
    out = ["VERSION 5.8 ;", 'DIVIDERCHAR "/" ;', 'BUSBITCHARS "[]" ;', f"DESIGN {net.top} ;",
           f"UNITS DISTANCE MICRONS {DBU} ;", f"DIEAREA ( 0 0 ) ( {d(pl.die[2])} {d(pl.die[3])} ) ;"]
    for r in range(rows):
        out.append(f"ROW ROW_{r} {site} {d(pl.core[0])} {d(pl.core[1] + r * rh)} {'FS' if r % 2 else 'N'} "
                   f"DO {nsite} BY 1 STEP {d(sw)} 0 ;")
    name = lambda s: re.sub(r"[^A-Za-z0-9_\[\].$/]", "_", s)
    out.append(f"COMPONENTS {len(net.cells)} ;")
    for c, (inst, typ) in enumerate(net.cells):
        out.append(f"- {name(inst)} {typ} + PLACED ( {d(pl.x[c])} {d(pl.y[c])} ) {'FS' if pl.flip[c] else 'N'} ;")
    out.append("END COMPONENTS")
    pins = [(p, b, dr) for p, (dr, ns) in net.ports.items() for b in range(len(ns))]
    out.append(f"PINS {len(pins)} ;")
    for p, b, dr in pins:
        x, y = pl.io[(p, b)]
        nm = f"{p}[{b}]" if len(net.ports[p][1]) > 1 else p
        out.append(f"- {nm} + NET {nm} + DIRECTION {dr.upper()} + USE SIGNAL + LAYER metal3 ( -70 -70 ) ( 70 70 ) "
                   f"+ PLACED ( {d(x)} {d(y)} ) N ;")
    out += ["END PINS", "END DESIGN", ""]
    with open(path, "w") as f:
        f.write("\n".join(out))


def _group(inst):
    m = re.search(r"row\[(\d+)\]\.col\[(\d+)\]", inst) or re.search(r"lane\[(\d+)\]()", inst)
    return m.groups() if m else None


def write_svg(path, net, pl, lib, critical=None, scale=6.0):
    W, H = pl.die[2] * scale, pl.die[3] * scale
    keys = sorted({k for k in (_group(i) for i, _ in net.cells) if k})
    hue = {k: i / max(1, len(keys)) for i, k in enumerate(keys)}
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}">',
         f'<rect width="{W:.0f}" height="{H:.0f}" fill="#f4f4f6"/>',
         f'<rect x="{pl.core[0]*scale:.1f}" y="{(pl.die[3]-pl.core[3])*scale:.1f}" width="{(pl.core[2]-pl.core[0])*scale:.1f}" '
         f'height="{(pl.core[3]-pl.core[1])*scale:.1f}" fill="#ffffff" stroke="#9aa0ad"/>']
    for c, (inst, typ) in enumerate(net.cells):
        k = _group(inst)
        r, g, b = colorsys.hls_to_rgb(hue[k], 0.45 if lib[typ].ff else 0.62, 0.55) if k else (0.6, 0.6, 0.62)
        s.append(f'<rect x="{pl.x[c]*scale:.1f}" y="{(pl.die[3]-pl.y[c]-pl.h[c])*scale:.1f}" width="{pl.w[c]*scale:.1f}" '
                 f'height="{pl.h[c]*scale:.1f}" fill="rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})" stroke="#ffffff" stroke-width="0.4">'
                 f'<title>{escape(inst)} ({typ})</title></rect>')
    for (p, b), (x, y) in pl.io.items():
        s.append(f'<circle cx="{x*scale:.1f}" cy="{(pl.die[3]-y)*scale:.1f}" r="2" fill="#2b2740"/>')
    if critical:
        pts = " ".join(f"{(pl.x[c]+pl.w[c]/2)*scale:.1f},{(pl.die[3]-pl.y[c]-pl.h[c]/2)*scale:.1f}" for c in critical)
        s.append(f'<polyline points="{pts}" fill="none" stroke="#d1242f" stroke-width="2.5" stroke-linejoin="round"/>')
    s.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(s))


def write_congestion_svg(path, grid, scale=6.0):
    g = grid.g
    util = np.zeros((grid.ny, grid.nx))
    util[:, :-1] = np.maximum(util[:, :-1], grid.use_h / grid.cap_h)
    util[:, 1:] = np.maximum(util[:, 1:], grid.use_h / grid.cap_h)
    util[:-1, :] = np.maximum(util[:-1, :], grid.use_v / grid.cap_v)
    util[1:, :] = np.maximum(util[1:, :], grid.use_v / grid.cap_v)
    W, H = grid.nx * g * scale, grid.ny * g * scale
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}">']
    for y in range(grid.ny):
        for x in range(grid.nx):
            u = min(util[y, x], 1.2) / 1.2
            r, gg, b = (int(40 + 215 * u), int(60 + 120 * (1 - abs(u - 0.5) * 2)), int(200 * (1 - u)))
            s.append(f'<rect x="{x*g*scale:.1f}" y="{(grid.ny-1-y)*g*scale:.1f}" width="{g*scale:.1f}" height="{g*scale:.1f}" '
                     f'fill="rgb({r},{gg},{b})"><title>{util[y, x]:.2f}</title></rect>')
    s.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(s))
