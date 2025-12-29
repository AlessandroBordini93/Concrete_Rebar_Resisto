# app.py — RebarCA API (Metodo A: ZIP “al volo”, niente disco persistente)
# MOD (ultima miglioria):
# - DISTF = "finestra gonfiata", ma la gonfiatura viene CLAMPATA (tagliata) per rimanere
#   SEMPRE dentro lo specchio murario del pannello (tra facce interne di pilastri e travi).
#   Quindi: dove posso aggiungo DISTF; dove non posso aggiungo il massimo possibile.
#
# Rimane invariato:
# - JSON input/output, endpoint, payload, results.stats + results.stats_table, ecc.
# - Filtro globale SOLO X per Xfin; Yfin senza filtro globale
# - Border-safe per ok_seg + clipping
# - Prune “trattini” V/H

from __future__ import annotations

import math
import bisect
import textwrap
import re
import zipfile
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, deque
from pathlib import Path
from io import BytesIO

# --- headless matplotlib (Render / server) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import RootModel

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

import pandas as pd

# optional
try:
    import ezdxf
except ImportError:
    ezdxf = None

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    PdfReader = PdfWriter = None


# ============================================================
# CONFIG
# ============================================================
FONT_REG, FONT_BOLD = "Helvetica", "Helvetica-Bold"
app = FastAPI(
    title="RebarCA API",
    version="2.3 (zip-on-the-fly, stats_table, x-only-filter, border-safe, window-real-validate, distf-clamped-to-panel)",
)

class Payload(RootModel[Dict[str, Any]]):
    pass


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class Beam:
    y_axis: float
    spess: float

@dataclass
class Column:
    x_axis: float
    spess: float

@dataclass
class Window:
    x: float
    y_rel: float
    w: float
    h: float
    y_abs: float = 0.0


# ============================================================
# VALIDATION + NAMING
# ============================================================
def _must(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)

def slugify(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "x"

def safe_suffix(s: str) -> str:
    s = (s or "").strip()
    _must(len(s) > 0, "meta.suffix mancante o vuoto")
    _must(len(s) <= 64, "meta.suffix troppo lungo (max 64)")
    _must(re.fullmatch(r"[A-Za-z0-9._-]+", s) is not None, "meta.suffix non valido (usa solo A-Z a-z 0-9 . _ -)")
    return s

ORIENT_MAP = {
    "n": "Nord", "nord": "Nord",
    "s": "Sud", "sud": "Sud",
    "e": "Est", "est": "Est",
    "o": "Ovest", "ovest": "Ovest", "w": "Ovest",
    "ne": "Nord-Est", "nordest": "Nord-Est", "nord-est": "Nord-Est",
    "no": "Nord-Ovest", "nordovest": "Nord-Ovest", "nord-ovest": "Nord-Ovest",
    "se": "Sud-Est", "sudest": "Sud-Est", "sud-est": "Sud-Est",
    "so": "Sud-Ovest", "sudovest": "Sud-Ovest", "sud-ovest": "Sud-Ovest",
}
ALLOWED_ORIENTATIONS = {"Nord", "Sud", "Est", "Ovest", "Nord-Est", "Nord-Ovest", "Sud-Est", "Sud-Ovest"}

def normalize_orientation(s: str) -> str:
    raw = (s or "").strip().lower().replace(" ", "")
    raw = raw.replace("–", "-").replace("—", "-").replace("_", "-")
    raw_no_dash = raw.replace("-", "")
    norm = ORIENT_MAP.get(raw_no_dash)
    if norm is not None:
        return norm
    s2 = (s or "").strip().replace("–", "-").replace("—", "-")
    if s2 in ALLOWED_ORIENTATIONS:
        return s2
    raise ValueError(f"meta.wall_orientation non valida. Valori ammessi: {sorted(ALLOWED_ORIENTATIONS)}")

def make_job_id(project_name: str, location_name: str, wall_orientation: str, suffix: str) -> Tuple[str, str]:
    o = normalize_orientation(wall_orientation)
    job_id = f"{slugify(project_name)}__{slugify(location_name)}__{slugify(o)}__{safe_suffix(suffix)}"
    return job_id, o


# ============================================================
# PDF HELPERS
# ============================================================
def _wrap(txt: str, width=92) -> str:
    return "\n".join(textwrap.fill(p.strip(), width) for p in txt.strip().splitlines())

def _footer(c: canvas.Canvas, W, H):
    h5 = H / 15
    c.setFont(FONT_REG, 11)
    c.drawCentredString(W / 2, 0.75 * h5, "Ing. Alessandro Bordini")
    c.drawCentredString(W / 2, 0.35 * h5, "Phone: 3451604706 - ✉: alessandro_bordini@outlook.com")


# ============================================================
# GEOMETRY UTILITIES
# ============================================================
def win_box(w: Window, pad: float = 0.0) -> Tuple[float, float, float, float]:
    return (w.x - pad, w.y_abs - pad, w.x + w.w + pad, w.y_abs + w.h + pad)

def _box_intersect_strict(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], eps: float = 1e-9) -> bool:
    # "strict": contatto sul bordo NON conta come intersezione
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 + eps or ax1 >= bx2 - eps or ay2 <= by1 + eps or ay1 >= by2 - eps)

def ok_seg(x1, y1, x2, y2, wins: List[Window], *, DISTF: float = 0.0, eps: float = 1e-9) -> bool:
    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))
    seg_box = (xmin, ymin, xmax, ymax)
    for w in wins:
        if _box_intersect_strict(seg_box, win_box(w, pad=DISTF), eps=eps):
            return False
    return True

def primarie(nodes, *, vertical: bool, PASSO: int, CLEAR: int):
    lo = nodes[0].x_axis if vertical else nodes[0].y_axis
    hi = nodes[-1].x_axis if vertical else nodes[-1].y_axis
    low = lo - nodes[0].spess / 2 + CLEAR
    high = hi + nodes[-1].spess / 2 - CLEAR

    best, full = 1e9, []
    for z0 in range(int(low), int(low) + PASSO):
        s = ((low - z0 + PASSO - 1) // PASSO) * PASSO + z0
        e = ((high - z0) // PASSO) * PASSO + z0
        if s > e:
            continue
        g = list(range(int(s), int(e) + 1, PASSO))

        if any(
            not any(a - n.spess / 2 + CLEAR <= v <= a + n.spess / 2 - CLEAR for v in g)
            for n, a in (((n, n.x_axis) if vertical else (n, n.y_axis)) for n in nodes)
        ):
            continue

        scrt = (g[0] - low) + (high - g[-1])
        if scrt < best:
            best, full = scrt, g

    if not full:
        raise ValueError("Maglia primaria da passo fisso impossibile (controlla geometria/spessori/CLEAR).")

    base = []
    for n in nodes:
        a, sp = (n.x_axis, n.spess) if vertical else (n.y_axis, n.spess)
        base.append(min((v for v in full if a - sp / 2 + CLEAR <= v <= a + sp / 2 - CLEAR), key=lambda v: abs(v - a)))
    return full, base


# ============================================================
# DISTF CLAMPED (finestra gonfiata ma sempre dentro pannello)
# ============================================================
def inflate_window_clamped_to_panel(
    w: Window,
    i: int,
    j: int,
    cols: List[Column],
    beams: List[Beam],
    DISTF: float,
    eps: float = 1e-9,
) -> Window:
    if DISTF <= 0:
        return w

    # facce interne pannello (specchio murario)
    xmin = cols[j].x_axis + cols[j].spess / 2.0
    xmax = cols[j + 1].x_axis - cols[j + 1].spess / 2.0
    ymin = beams[i].y_axis + beams[i].spess / 2.0
    ymax = beams[i + 1].y_axis - beams[i + 1].spess / 2.0

    xa = w.x
    ya = w.y_abs
    xb = w.x + w.w
    yb = w.y_abs + w.h

    # quanto "spazio" ho prima di uscire dal pannello
    pad_l = max(0.0, min(DISTF, xa - xmin))
    pad_r = max(0.0, min(DISTF, xmax - xb))
    pad_b = max(0.0, min(DISTF, ya - ymin))
    pad_t = max(0.0, min(DISTF, ymax - yb))

    # se per qualche motivo siamo quasi sul bordo (numerica), clampo duro
    if xa - pad_l < xmin - eps:
        pad_l = max(0.0, xa - xmin)
    if xb + pad_r > xmax + eps:
        pad_r = max(0.0, xmax - xb)
    if ya - pad_b < ymin - eps:
        pad_b = max(0.0, ya - ymin)
    if yb + pad_t > ymax + eps:
        pad_t = max(0.0, ymax - yb)

    return Window(
        x=xa - pad_l,
        y_rel=w.y_rel,  # non usato nel calcolo geometrico (solo info)
        w=w.w + pad_l + pad_r,
        h=w.h + pad_b + pad_t,
        y_abs=ya - pad_b,
    )

def build_inflated_windows_clamped(
    win_data: Dict[Tuple[int, int], List[Window]],
    cols: List[Column],
    beams: List[Beam],
    DISTF: float,
) -> List[Window]:
    out: List[Window] = []
    for (i, j), lst in win_data.items():
        for w in lst:
            out.append(inflate_window_clamped_to_panel(w, i=i, j=j, cols=cols, beams=beams, DISTF=DISTF))
    return out


def linee_finestre(grid: List[float], wins_gonf: List[Window], asse: str) -> List[float]:
    """
    Regola:
      - asse="x": scelgo sx/dx per ogni finestra gonfiata, SKIPPANDO candidati che cadono
        dentro QUALSIASI finestra gonfiata (filtro globale X).
      - asse="y": scelgo giù/su per ogni finestra gonfiata, SENZA filtro globale.
    """
    grid = sorted(grid)

    def ok_axis_value_global_x(x: float) -> bool:
        for w in wins_gonf:
            if w.x < x < (w.x + w.w):  # strict inside
                return False
        return True

    extra: List[float] = []
    for w in wins_gonf:
        if asse == "x":
            L = w.x
            R = w.x + w.w

            i_sx = bisect.bisect_right(grid, L) - 1
            while i_sx >= 0 and not ok_axis_value_global_x(grid[i_sx]):
                i_sx -= 1

            i_dx = bisect.bisect_left(grid, R)
            while i_dx < len(grid) and not ok_axis_value_global_x(grid[i_dx]):
                i_dx += 1

            if i_sx >= 0:
                extra.append(grid[i_sx])
            if i_dx < len(grid):
                extra.append(grid[i_dx])

        else:
            B = w.y_abs
            T = w.y_abs + w.h

            i_giu = bisect.bisect_right(grid, B) - 1
            i_su = bisect.bisect_left(grid, T)

            if i_giu >= 0:
                extra.append(grid[i_giu])
            if i_su < len(grid):
                extra.append(grid[i_su])

    return sorted(set(extra))

def intermedie(lines: List[float], PASSO: int) -> List[float]:
    out: List[float] = []
    for a, b in zip(lines[:-1], lines[1:]):
        rem = b - a
        if rem < 2 * PASSO:
            continue
        pos = a
        while rem > PASSO:
            step = 100 if rem >= 100 else 75 if rem >= 75 else 50 if rem >= 50 else PASSO
            if rem == 125:
                step = 75
            pos += step
            out.append(pos)
            rem = b - pos
    return out

def _merge_intervals(ints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not ints:
        return []
    ints = sorted((min(a, b), max(a, b)) for a, b in ints)
    out = [ints[0]]
    for a, b in ints[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out

def _subtract_intervals(base: Tuple[float, float], cuts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    a, b = min(base), max(base)
    cuts = _merge_intervals([(max(a, c1), min(b, c2)) for c1, c2 in cuts if not (c2 <= a or c1 >= b)])
    if not cuts:
        return [(a, b)]
    out: List[Tuple[float, float]] = []
    cur = a
    for c1, c2 in cuts:
        if c1 > cur:
            out.append((cur, c1))
        cur = max(cur, c2)
    if cur < b:
        out.append((cur, b))
    return [(x1, x2) for x1, x2 in out if x2 > x1]

def clip_vertical_segment(x: float, y1: float, y2: float, wins: List[Window], DISTF: float, eps: float = 1e-9) -> List[Tuple[float, float]]:
    # border-safe: taglio solo se la linea passa "dentro" (strict) la proiezione X della finestra gonfiata
    ya, yb = min(y1, y2), max(y1, y2)
    cuts: List[Tuple[float, float]] = []
    for w in wins:
        xmin, ymin, xmax, ymax = win_box(w, pad=DISTF)
        if (xmin + eps) < x < (xmax - eps):
            cuts.append((ymin, ymax))
    return _subtract_intervals((ya, yb), cuts)

def clip_horizontal_segment(y: float, x1: float, x2: float, wins: List[Window], DISTF: float, eps: float = 1e-9) -> List[Tuple[float, float]]:
    # border-safe: taglio solo se la linea passa "dentro" (strict) la proiezione Y della finestra gonfiata
    xa, xb = min(x1, x2), max(x1, x2)
    cuts: List[Tuple[float, float]] = []
    for w in wins:
        xmin, ymin, xmax, ymax = win_box(w, pad=DISTF)
        if (ymin + eps) < y < (ymax - eps):
            cuts.append((xmin, xmax))
    return _subtract_intervals((xa, xb), cuts)


# ============================================================
# PRUNE “TRATTINI” (dangling)
# ============================================================
def _k(v: float, nd: int = 6) -> float:
    return round(float(v), nd)

def split_segments_at_intersections(
    v_segs: List[Tuple[float, float, float]],
    h_segs: List[Tuple[float, float, float]],
    nd: int = 6,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float], str]], Dict[Tuple[float, float], set]]:
    v_norm = [(_k(x, nd), _k(min(y1, y2), nd), _k(max(y1, y2), nd)) for x, y1, y2 in v_segs if max(y1, y2) > min(y1, y2)]
    h_norm = [(_k(y, nd), _k(min(x1, x2), nd), _k(max(x1, x2), nd)) for y, x1, x2 in h_segs if max(x1, x2) > min(x1, x2)]

    v_points: Dict[Tuple[float, float, float], set] = {}
    h_points: Dict[Tuple[float, float, float], set] = {}

    for x, y1, y2 in v_norm:
        v_points[(x, y1, y2)] = {y1, y2}
    for y, x1, x2 in h_norm:
        h_points[(y, x1, x2)] = {x1, x2}

    def between(a, b, x):
        return a <= x <= b

    for x, y1, y2 in v_norm:
        for y, x1, x2 in h_norm:
            if between(x1, x2, x) and between(y1, y2, y):
                v_points[(x, y1, y2)].add(y)
                h_points[(y, x1, x2)].add(x)

    edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []
    adj: Dict[Tuple[float, float], set] = defaultdict(set)

    def add_edge(u, v, typ):
        if u == v:
            return
        eid = len(edges)
        edges.append((u, v, typ))
        adj[u].add(eid)
        adj[v].add(eid)

    for (x, y1, y2), ys in v_points.items():
        ys2 = sorted(ys)
        for a, b in zip(ys2[:-1], ys2[1:]):
            add_edge((x, a), (x, b), "v")

    for (y, x1, x2), xs in h_points.items():
        xs2 = sorted(xs)
        for a, b in zip(xs2[:-1], xs2[1:]):
            add_edge((a, y), (b, y), "h")

    return edges, adj

def prune_dangling(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]],
    adj: Dict[Tuple[float, float], set],
    protected_nodes: set,
) -> List[bool]:
    alive = [True] * len(edges)

    def degree(node):
        return sum(1 for eid in adj.get(node, set()) if alive[eid])

    q = deque([n for n in adj.keys() if degree(n) == 1 and n not in protected_nodes])

    while q:
        n = q.popleft()
        if n in protected_nodes:
            continue
        if degree(n) != 1:
            continue
        eids = [eid for eid in adj[n] if alive[eid]]
        if not eids:
            continue
        eid = eids[0]
        alive[eid] = False
        u, v, _t = edges[eid]
        other = v if u == n else u
        if other not in protected_nodes and degree(other) == 1:
            q.append(other)

    return alive

def merge_atomic_edges(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]],
    alive: List[bool],
    nd: int = 6,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    v_parts = []
    h_parts = []
    for ok, (u, v, t) in zip(alive, edges):
        if not ok:
            continue
        (x1, y1), (x2, y2) = u, v
        if t == "v":
            x = x1
            ylo, yhi = sorted((y1, y2))
            v_parts.append((_k(x, nd), _k(ylo, nd), _k(yhi, nd)))
        else:
            y = y1
            xlo, xhi = sorted((x1, x2))
            h_parts.append((_k(y, nd), _k(xlo, nd), _k(xhi, nd)))

    v_by_x = defaultdict(list)
    for x, y1, y2 in v_parts:
        v_by_x[x].append((y1, y2))
    v_out = []
    for x, lst in v_by_x.items():
        lst = sorted(lst)
        cur_a, cur_b = lst[0]
        for a, b in lst[1:]:
            if a <= cur_b:
                cur_b = max(cur_b, b)
            else:
                v_out.append((x, cur_a, cur_b))
                cur_a, cur_b = a, b
        v_out.append((x, cur_a, cur_b))

    h_by_y = defaultdict(list)
    for y, x1, x2 in h_parts:
        h_by_y[y].append((x1, x2))
    h_out = []
    for y, lst in h_by_y.items():
        lst = sorted(lst)
        cur_a, cur_b = lst[0]
        for a, b in lst[1:]:
            if a <= cur_b:
                cur_b = max(cur_b, b)
            else:
                h_out.append((y, cur_a, cur_b))
                cur_a, cur_b = a, b
        h_out.append((y, cur_a, cur_b))

    return v_out, h_out

def prune_vh_segments(
    Xall: List[float],
    Yall: List[float],
    Xbase: List[float],
    Ybase: List[float],
    v_segs: List[Tuple[float, float, float]],
    h_segs: List[Tuple[float, float, float]],
    nd: int = 6,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    protected = set()
    Xbase_set = {_k(x, nd) for x in Xbase}
    Ybase_set = {_k(y, nd) for y in Ybase}
    xmin, xmax = _k(min(Xall), nd), _k(max(Xall), nd)
    ymin, ymax = _k(min(Yall), nd), _k(max(Yall), nd)
    for x in Xall:
        for y in Yall:
            xx, yy = _k(x, nd), _k(y, nd)
            if (xx in Xbase_set) or (yy in Ybase_set) or (xx in (xmin, xmax)) or (yy in (ymin, ymax)):
                protected.add((xx, yy))
    edges, adj = split_segments_at_intersections(v_segs, h_segs, nd=nd)
    alive = prune_dangling(edges, adj, protected_nodes=protected)
    v2, h2 = merge_atomic_edges(edges, alive, nd=nd)
    return v2, h2


# ============================================================
# STIFFNESS
# ============================================================
def diagonali_rigidezze(
    Xall: List[float],
    Yall: List[float],
    cols: List[Column],
    beams: List[Beam],
    wins: List[Window],
    *,
    EA: float,
    CLEAR: int,
    DISTF: int,
) -> Dict[Tuple[int, int], List[List[float]]]:
    Xstr = [x for x in Xall if any(abs(x - c.x_axis) <= c.spess / 2 - CLEAR + 1e-6 for c in cols)]
    Ystr = [y for y in Yall if any(abs(y - b.y_axis) <= b.spess / 2 - CLEAR + 1e-6 for b in beams)]
    Xstr.sort()
    Ystr.sort()

    pannelli = defaultdict(list)
    for ix in range(len(Xall) - 1):
        for iy in range(len(Yall) - 1):
            x1, x2 = Xall[ix], Xall[ix + 1]
            y1, y2 = Yall[iy], Yall[iy + 1]
            if not ok_seg(x1, y1, x2, y2, wins, DISTF=DISTF):
                continue
            j = bisect.bisect_right(Xstr, x1) - 1
            i = bisect.bisect_right(Ystr, y1) - 1
            b_h = x2 - x1
            L = math.hypot(x2 - x1, y2 - y1)
            k = EA * (b_h * 10) / ((L * 10) * (L * 10))
            pannelli[(i, j)].append((y1, x1, k))

    rig: Dict[Tuple[int, int], List[List[float]]] = {}
    for (i, j), diag in pannelli.items():
        if not diag:
            rig[(i, j)] = [[0.0]]
            continue
        diag.sort(key=lambda t: (t[0], t[1]))
        cols_ref = sorted({x for _, x, _ in diag})
        rows_ref = sorted({y for y, _, _ in diag})
        mat = [[0.0] * len(cols_ref) for _ in range(len(rows_ref))]
        c_idx = {x: ii for ii, x in enumerate(cols_ref)}
        r_idx = {y: ii for ii, y in enumerate(rows_ref)}
        for y1, x1, k in diag:
            mat[r_idx[y1]][c_idx[x1]] = k
        rig[(i, j)] = mat

    for i in range(len(beams) - 1):
        for j in range(len(cols) - 1):
            rig.setdefault((i, j), [[0.0]])
    return rig


# ============================================================
# PDF GENERATORS
# ============================================================
def _first_page(schema_png: Path, stats: List[str], out_pdf: Path, header_lines: List[str]):
    W, H = A4
    h1, h3 = H / 6, 3 * H / 6
    c = canvas.Canvas(str(out_pdf), pagesize=A4)

    c.setFont(FONT_BOLD, 14)
    c.drawCentredString(W / 2, H - 0.75 * h1, "Calcolo Automatizzato – Schema di posa Resisto 5.9")

    c.setFont(FONT_REG, 11)
    y = H - h1 + 0.3 * cm
    for ln in header_lines:
        c.drawCentredString(W / 2, y, ln)
        y -= 0.45 * cm

    img = ImageReader(str(schema_png))
    iw, ih = img.getSize()
    h_img = min(h3 - 1 * cm, ih)
    w_img = h_img * iw / ih
    if w_img > W - 2 * cm:
        w_img = W - 2 * cm
        h_img = w_img * ih / iw

    c.drawImage(img, (W - w_img) / 2, H - h1 - h_img - 0.5 * cm, w_img, h_img)

    c.setFont(FONT_REG, 10)
    y0 = H - h1 - h3 - 0.5 * cm
    c.drawString(2 * cm, y0, "======================== STATISTICHE RINFORZO ========================")
    y = y0 - 0.45 * cm
    for ln in stats:
        c.drawString(2 * cm, y, ln)
        y -= 0.40 * cm
    c.drawString(2 * cm, y, "======================================================================")

    _footer(c, W, H)
    c.save()

def _extra_pages(
    matrices: Dict[Tuple[int, int], "pd.DataFrame"],
    Aeq: Dict[Tuple[int, int], float],
    Keq: Dict[Tuple[int, int], float],
    grafico1: Path,
    grafico2: Path,
    area_uni: float,
    out_pdf: Path,
):
    W, H = A4
    MARGX = 2 * cm
    c = canvas.Canvas(str(out_pdf), pagesize=A4)

    def _draw_equations(y0: float) -> float:
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(MARGX, y0, "Calcolo rigidezze equivalenti – riepilogo formule")
        y = y0 - 12
        eqs = [
            r"$K_{d,i}= \dfrac{E\,A}{L_i^{2}}\,b_i$",
            r"$K_{\text{or}}= \dfrac{1}{\sum K_{d,i,x}}$",
            r"$K_{\text{eq}}= \dfrac{1}{\sum K_{\text{or}}}$",
            r"$A_{\text{eq}}= \dfrac{K_{\text{eq}}\,l^{2}}{E\,b}$",
        ]
        for eq in eqs:
            buf = BytesIO()
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, eq, fontsize=6)
            fig.patch.set_alpha(0)
            plt.axis("off")
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
            plt.close(fig)
            buf.seek(0)
            img = ImageReader(buf)
            iw, ih = img.getSize()
            scale = 0.5
            c.drawImage(img, MARGX, y - ih * scale, iw * scale, ih * scale, mask="auto")
            y -= ih * scale + 2
        return y

    top_y = H - 1.6 * cm
    y = _draw_equations(top_y) - 50

    for (i, j) in sorted(matrices):
        df = matrices[(i, j)]
        lines = df.to_string().splitlines()
        blocco_h = (len(lines) + 3) * 0.32 * cm
        if y - blocco_h < 2 * cm:
            _footer(c, W, H)
            c.showPage()
            y = H - 2.5 * cm

        c.setFont(FONT_BOLD, 11)
        c.drawString(MARGX, y, f"Piano: {i} – Tamponamento: {j+1}")
        y -= 0.45 * cm

        c.setFont("Courier", 8)
        for ln in lines:
            c.drawString(MARGX, y, ln)
            y -= 0.32 * cm

        y -= 0.5 * cm
        c.setFont(FONT_BOLD, 9)
        c.setFillColor(colors.red)
        c.drawString(MARGX, y, f"Aeq = {Aeq[(i,j)]:.0f} mm² || Keq = {Keq[(i,j)]:.0f} N/mm")
        y -= 0.7 * cm
        c.setFillColor(colors.black)

    testo = _wrap(f"Aunivoca = {area_uni:.0f} mm²", 80)
    if y - 2 * cm < 2 * cm:
        _footer(c, W, H)
        c.showPage()
        y = H - 2.5 * cm

    c.setFont(FONT_BOLD, 11)
    c.setFillColor(colors.blue)
    c.drawString(MARGX, y, "Calcolo Area Equivalente univoca")
    y -= 0.45 * cm
    c.setFont(FONT_BOLD, 9)
    c.setFillColor(colors.red)
    for ln in testo.splitlines():
        c.drawString(MARGX, y, ln)
        y -= 0.32 * cm
    c.setFillColor(colors.black)

    _footer(c, W, H)
    c.showPage()

    img1, img2 = ImageReader(str(grafico1)), ImageReader(str(grafico2))
    iw1, ih1 = img1.getSize()
    iw2, ih2 = img2.getSize()
    slot_h = (H - 5 * cm) / 2

    def _place(img, iw, ih, y_top):
        h = min(slot_h, ih)
        w = h * iw / ih
        if w > W - 3 * cm:
            w = W - 3 * cm
            h = w * ih / iw
        c.drawImage(img, (W - w) / 2, y_top - h, w, h)

    c.setFont(FONT_BOLD, 13)
    c.drawCentredString(W / 2, H - 1.5 * cm, "Grafici – Diagonali Equivalenti")
    _place(img1, iw1, ih1, H - 2.7 * cm)
    _place(img2, iw2, ih2, H - 2.7 * cm - slot_h - 0.8 * cm)
    _footer(c, W, H)
    c.save()


# ============================================================
# DXF EXPORT
# ============================================================
def _export_dxf(
    cols: List[Column],
    beams: List[Beam],
    finestre_real: List[Window],
    finestre_infl: List[Window],
    X: List[float],
    Y: List[float],
    *,
    Xbase: List[float],
    Ybase: List[float],
    path: Path,
) -> bool:
    if ezdxf is None:
        return False

    doc = ezdxf.new(setup=True)
    m = doc.modelspace()
    doc.layers.new("Struttura", dxfattribs={"color": 1})
    doc.layers.new("Resisto", dxfattribs={"color": 7})

    def rect(x1, y1, x2, y2):
        m.add_lwpolyline([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], dxfattribs={"layer": "Struttura"})

    y_min = beams[0].y_axis - beams[0].spess / 2
    y_max = beams[-1].y_axis + beams[-1].spess / 2
    for c in cols:
        rect(c.x_axis - c.spess / 2, y_min, c.x_axis + c.spess / 2, y_max)

    x_min = cols[0].x_axis - cols[0].spess / 2
    x_max = cols[-1].x_axis + cols[-1].spess / 2
    for b in beams:
        rect(x_min, b.y_axis - b.spess / 2, x_max, b.y_axis + b.spess / 2)

    # disegno finestre REALI (non gonfiate)
    for w in finestre_real:
        rect(w.x, w.y_abs, w.x + w.w, w.y_abs + w.h)

    add = lambda a, b: m.add_line(a, b, dxfattribs={"layer": "Resisto"})

    v_raw: List[Tuple[float, float, float]] = []
    h_raw: List[Tuple[float, float, float]] = []

    # clipping su finestre gonfiate CLAMPATE (DISTF=0 perché già gonfiate)
    for x in X:
        for y1, y2 in zip(Y[:-1], Y[1:]):
            for ya, yb in clip_vertical_segment(x, y1, y2, finestre_infl, DISTF=0):
                if yb > ya:
                    v_raw.append((x, ya, yb))

    for y in Y:
        for x1, x2 in zip(X[:-1], X[1:]):
            for xa, xb in clip_horizontal_segment(y, x1, x2, finestre_infl, DISTF=0):
                if xb > xa:
                    h_raw.append((y, xa, xb))

    v_segs, h_segs = prune_vh_segments(X, Y, Xbase, Ybase, v_raw, h_raw, nd=6)

    for x, y1, y2 in v_segs:
        add((x, y1), (x, y2))
    for y, x1, x2 in h_segs:
        add((x1, y), (x2, y))

    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            a, b = X[i], X[i + 1]
            c_, d_ = Y[j], Y[j + 1]
            if ok_seg(a, c_, b, d_, finestre_infl, DISTF=0):
                add((a, c_), (b, d_))
            if ok_seg(a, d_, b, c_, finestre_infl, DISTF=0):
                add((a, d_), (b, c_))

    doc.saveas(str(path))
    return path.exists()


# ============================================================
# PARSE PAYLOAD
# ============================================================
def parse_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("meta", {})
    project_name = meta.get("project_name", "")
    location_name = meta.get("location_name", "")
    suffix = meta.get("suffix", "")
    wall_orientation_raw = meta.get("wall_orientation", "")

    _must(project_name.strip() != "", "meta.project_name mancante o vuoto")
    _must(location_name.strip() != "", "meta.location_name mancante o vuoto")
    job_id, wall_orientation = make_job_id(project_name, location_name, wall_orientation_raw, suffix)

    settings = payload.get("settings", {})
    PASSO = int(settings.get("PASSO", 25))
    CLEAR = int(settings.get("CLEAR", 5))
    DISTF = int(settings.get("DISTF", 10))
    E = float(settings.get("E_MPa", 210000))
    A = float(settings.get("A_mm2", 150))

    PASSO = max(1, PASSO)
    CLEAR = max(5, CLEAR)
    DISTF = max(0, DISTF)
    EA = E * A

    grid = payload["grid"]
    np_ = int(grid["np"])
    nt_ = int(grid["nt"])

    spB = list(map(float, grid["beams"]["spessori_cm"]))
    intB = list(map(float, grid["beams"]["interassi_cm"]))
    spC = list(map(float, grid["columns"]["spessori_cm"]))
    intC = list(map(float, grid["columns"]["interassi_cm"]))

    _must(np_ > 0 and nt_ > 0, "np/nt devono essere >0")
    _must(len(spB) == nt_ + 1, "beams.spessori_cm deve avere lunghezza nt+1")
    _must(len(intB) == nt_, "beams.interassi_cm deve avere lunghezza nt")
    _must(len(spC) == np_ + 1, "columns.spessori_cm deve avere lunghezza np+1")
    _must(len(intC) == np_, "columns.interassi_cm deve avere lunghezza np")
    _must(all(v > 0 for v in spB + intB + spC + intC), "spessori/interassi devono essere >0")

    y = [0.0]
    for v in intB:
        y.append(y[-1] + v)
    x = [0.0]
    for v in intC:
        x.append(x[-1] + v)

    beams = [Beam(yy, sp) for yy, sp in zip(y, spB)]
    cols = [Column(xx, sp) for xx, sp in zip(x, spC)]

    win_data: Dict[Tuple[int, int], List[Window]] = defaultdict(list)

    export = payload.get("export", {})
    export_png = bool(export.get("png", False))
    export_pdf = bool(export.get("pdf", False))
    export_dxf = bool(export.get("dxf", False))

    # validazione finestre reali nello specchio (NO DISTF, NO CLEAR)
    EPSG = 1e-9

    for item in payload.get("openings", []):
        i = int(item["panel"]["i"])
        j = int(item["panel"]["j"])
        _must(0 <= i < nt_, f"panel.i fuori range: {i}")
        _must(0 <= j < np_, f"panel.j fuori range: {j}")

        for w in item.get("windows", []):
            dx = float(w["dx_cm"])
            dy = float(w["dy_cm"])
            ww = float(w["w_cm"])
            hh = float(w["h_cm"])
            _must(ww > 0 and hh > 0, "w/h finestra devono essere >0")

            x_pil_sx = cols[j].x_axis
            x_pil_dx = cols[j + 1].x_axis
            y_trave_inf = beams[i].y_axis
            y_trave_sup = beams[i + 1].y_axis

            sp_pil_sx = cols[j].spess
            sp_pil_dx = cols[j + 1].spess
            sp_tr_inf = beams[i].spess
            sp_tr_sup = beams[i + 1].spess

            xa = x_pil_sx + dx
            ya = y_trave_inf + dy
            xb = xa + ww
            yb = ya + hh

            xmin = x_pil_sx + sp_pil_sx / 2.0
            xmax = x_pil_dx - sp_pil_dx / 2.0
            ymin = y_trave_inf + sp_tr_inf / 2.0
            ymax = y_trave_sup - sp_tr_sup / 2.0

            _must(
                not (xa < xmin - EPSG or xb > xmax + EPSG or ya < ymin - EPSG or yb > ymax + EPSG),
                f"Finestra esce dal pannello (i={i}, j={j}).",
            )

            win_data[(i, j)].append(Window(x=xa, y_rel=dy, w=ww, h=hh, y_abs=ya))

    return {
        "job_id": job_id,
        "meta_norm": {
            "project_name": project_name,
            "location_name": location_name,
            "suffix": suffix,
            "wall_orientation": wall_orientation,
        },
        "export_png": export_png,
        "export_pdf": export_pdf,
        "export_dxf": export_dxf,
        "PASSO": PASSO,
        "CLEAR": CLEAR,
        "DISTF": DISTF,
        "E": E,
        "A": A,
        "EA": EA,
        "np": np_,
        "nt": nt_,
        "beams": beams,
        "cols": cols,
        "win_data": dict(win_data),
    }


# ============================================================
# OVERLAY HELPERS
# ============================================================
def _line(a, b, layer, stroke="#111", width=1, dash=None, panel=None):
    return {
        "type": "line",
        "layer": layer,
        "a": [float(a[0]), float(a[1])],
        "b": [float(b[0]), float(b[1])],
        **({"panel": panel} if panel is not None else {}),
        "style": {"stroke": stroke, "width": width, "dash": dash or []},
    }

def _text(pos, text, layer="label", fill="#111", size=10, panel=None):
    return {
        "type": "text",
        "layer": layer,
        "pos": [float(pos[0]), float(pos[1])],
        "text": str(text),
        **({"panel": panel} if panel is not None else {}),
        "style": {"fill": fill, "size": size},
    }


# ============================================================
# COMPUTE CORE
# ============================================================
def compute(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = parse_payload(payload)
    PASSO, CLEAR, DISTF = cfg["PASSO"], cfg["CLEAR"], cfg["DISTF"]
    E, EA = cfg["E"], cfg["EA"]

    cols: List[Column] = cfg["cols"]
    beams: List[Beam] = cfg["beams"]
    win_data: Dict[Tuple[int, int], List[Window]] = cfg["win_data"]
    all_w_real = [w for lst in win_data.values() for w in lst]

    # finestre gonfiate CLAMPATE allo specchio murario del pannello
    all_w_infl = build_inflated_windows_clamped(win_data, cols, beams, DISTF)

    # primarie
    Xfull, Xbase = primarie(cols, vertical=True, PASSO=PASSO, CLEAR=CLEAR)
    Yfull, Ybase = primarie(beams, vertical=False, PASSO=PASSO, CLEAR=CLEAR)

    # linee vicino finestre: uso finestre gonfiate (già clampate)
    Xfin = linee_finestre(Xfull, all_w_infl, "x")
    Yfin = linee_finestre(Yfull, all_w_infl, "y")

    # filtro globale SOLO X
    EPS = 1e-9
    def _inside_any_x(x: float) -> bool:
        return any((w.x + EPS) < x < (w.x + w.w - EPS) for w in all_w_infl)

    Xfin = sorted(set(x for x in Xfin if not _inside_any_x(x)))
    Yfin = sorted(set(Yfin))

    # intermedie
    Xsec = intermedie(sorted(set(Xbase + Xfin)), PASSO=PASSO)
    Ysec = intermedie(sorted(set(Ybase + Yfin)), PASSO=PASSO)

    Xall = sorted(set(Xbase + Xfin + Xsec))
    Yall = sorted(set(Ybase + Yfin + Ysec))

    # V/H raw con clipping vs finestre gonfiate clampate (DISTF=0 perché sono già gonfiate)
    v_raw: List[Tuple[float, float, float]] = []
    h_raw: List[Tuple[float, float, float]] = []

    for y in Yall:
        for x1, x2 in zip(Xall[:-1], Xall[1:]):
            for xa, xb in clip_horizontal_segment(y, x1, x2, all_w_infl, DISTF=0):
                if xb > xa:
                    h_raw.append((y, xa, xb))

    for x in Xall:
        for y1, y2 in zip(Yall[:-1], Yall[1:]):
            for ya, yb in clip_vertical_segment(x, y1, y2, all_w_infl, DISTF=0):
                if yb > ya:
                    v_raw.append((x, ya, yb))

    # prune trattini
    v_segs, h_segs = prune_vh_segments(Xall, Yall, Xbase, Ybase, v_raw, h_raw, nd=6)

    # stats (V/H pruned, diagonali)
    Lh_cm = sum(x2 - x1 for (y, x1, x2) in h_segs)
    Lv_cm = sum(y2 - y1 for (x, y1, y2) in v_segs)
    n_o = len(h_segs)
    n_v = len(v_segs)

    Ld_cm = 0.0
    n_d = 0
    for i in range(len(Xall) - 1):
        for j in range(len(Yall) - 1):
            a, b = Xall[i], Xall[i + 1]
            c_, d_ = Yall[j], Yall[j + 1]
            dlen = math.hypot(b - a, d_ - c_)
            if ok_seg(a, c_, b, d_, all_w_infl, DISTF=0):
                n_d += 1
                Ld_cm += dlen
            if ok_seg(a, d_, b, c_, all_w_infl, DISTF=0):
                n_d += 1
                Ld_cm += dlen

    width_cm = cols[-1].x_axis
    height_cm = beams[-1].y_axis - beams[0].y_axis
    area_tot_m2 = (width_cm * height_cm) / 10_000
    area_open_m2 = sum(w.w * w.h for w in all_w_real) / 10_000
    area_pieno_m2 = max(area_tot_m2 - area_open_m2, 1e-6)

    def inc(n, a):
        return n / 2.75 / a

    p_medio = ((Lv_cm / n_v) + (Lh_cm / n_o)) / 2 if (n_v and n_o) else 0.0

    stats = [
        f"Orizzontali : L = {Lh_cm/100:.2f} m | n = {n_o} | Inc. P {inc(n_o,area_pieno_m2):.2f} T {inc(n_o,area_tot_m2):.2f}",
        f"Verticali : L = {Lv_cm/100:.2f} m | n = {n_v} | Inc. P {inc(n_v,area_pieno_m2):.2f} T {inc(n_v,area_tot_m2):.2f}",
        f"Diagonali : L = {Ld_cm/100:.2f} m | n = {n_d} --> 2 x n°: {n_d/2:.0f} | Inc. P {inc(n_d,area_pieno_m2):.2f} T {inc(n_d,area_tot_m2):.2f}",
        "======================================================================",
        f"Passo medio = {p_medio:.1f} cm",
        f"Area pieno (senza aperture) : {area_pieno_m2:.2f} m²",
        f"Area totale : {area_tot_m2:.2f} m²",
    ]

    stats_table = {
        "orizzontali": {"L_m": float(round(Lh_cm / 100.0, 4)), "n": int(n_o), "inc_p": float(round(inc(n_o, area_pieno_m2), 4)), "inc_t": float(round(inc(n_o, area_tot_m2), 4))},
        "verticali": {"L_m": float(round(Lv_cm / 100.0, 4)), "n": int(n_v), "inc_p": float(round(inc(n_v, area_pieno_m2), 4)), "inc_t": float(round(inc(n_v, area_tot_m2), 4))},
        "diagonali": {"L_m": float(round(Ld_cm / 100.0, 4)), "n": int(n_d), "n_x2": float(round(n_d / 2.0, 4)), "inc_p": float(round(inc(n_d, area_pieno_m2), 4)), "inc_t": float(round(inc(n_d, area_tot_m2), 4))},
        "passo_medio_cm": float(round(p_medio, 3)),
        "area_pieno_m2": float(round(area_pieno_m2, 6)),
        "area_tot_m2": float(round(area_tot_m2, 6)),
    }

    # stiffness: diagonali_rigidezze usa ok_seg(..., DISTF=DISTF) internamente (pad),
    # ma noi vogliamo finestre già gonfiate+clampate: passiamo DISTF=0 e wins=infl
    rig = diagonali_rigidezze(Xall, Yall, cols, beams, all_w_infl, EA=EA, CLEAR=CLEAR, DISTF=0)

    Aeq_dict: Dict[Tuple[int, int], float] = {}
    Keq_dict: Dict[Tuple[int, int], float] = {}
    Kad_dict: Dict[Tuple[int, int], float] = {}
    matrices_for_pdf: Dict[Tuple[int, int], "pd.DataFrame"] = {}

    for (i, j), mat in rig.items():
        df = pd.DataFrame(reversed(mat)).round(0).astype(int)
        df_tmp = df.copy()
        df_tmp["Kor"] = df_tmp.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
        K_eq = 1 / df_tmp["Kor"].sum() if df_tmp["Kor"].sum() else 0.0

        Lx = cols[j + 1].x_axis - cols[j].x_axis
        Ly = beams[i + 1].y_axis - beams[i].y_axis
        Ld_mm = math.hypot(Lx * 10, Ly * 10)
        Aeq = (K_eq * (Ld_mm**2) / (Lx * 10) / E) if (K_eq > 0 and Lx > 0) else 0.0
        Kad = ((E * (Lx * 10)) / (Ld_mm**2)) if Ld_mm > 0 else 0.0

        Aeq_dict[(i, j)] = float(round(Aeq, 2))
        Keq_dict[(i, j)] = float(round(K_eq, 2))
        Kad_dict[(i, j)] = float(round(Kad, 2))

        df_pdf = df.copy()
        df_pdf["Kor"] = df_tmp["Kor"].map(lambda v: f"{v:.2e}")
        matrices_for_pdf[(i, j)] = df_pdf

    # univoca
    s_keq = pd.Series(Keq_dict, name="Keq")
    s_keq.index = pd.MultiIndex.from_tuples(s_keq.index, names=["i_trave", "j_pilastro"])
    df_keq = s_keq.unstack(level="j_pilastro").sort_index(axis=0, ascending=False).fillna(0.0)
    df_keq["Kor"] = df_keq.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
    K_eq_u = 1 / df_keq["Kor"].sum() if df_keq["Kor"].sum() else 0.0

    s_kad = pd.Series(Kad_dict, name="Kad")
    s_kad.index = pd.MultiIndex.from_tuples(s_kad.index, names=["i_trave", "j_pilastro"])
    df_kad = s_kad.unstack(level="j_pilastro").sort_index(axis=0, ascending=False).fillna(0.0)
    df_kad["Kor"] = df_kad.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
    K_eq_u_adi = 1 / df_kad["Kor"].sum() if df_kad["Kor"].sum() else 0.0

    Aeq_univoca = (K_eq_u / K_eq_u_adi) if K_eq_u_adi else 0.0

    # overlays
    grid_entities = []
    for x, y1, y2 in v_segs:
        grid_entities.append(_line((x, y1), (x, y2), layer="grid_v"))
    for y, x1, x2 in h_segs:
        grid_entities.append(_line((x1, y), (x2, y), layer="grid_h"))
    for ii in range(len(Xall) - 1):
        for jj in range(len(Yall) - 1):
            a, b = Xall[ii], Xall[ii + 1]
            c_, d_ = Yall[jj], Yall[jj + 1]
            if ok_seg(a, c_, b, d_, all_w_infl, DISTF=0):
                grid_entities.append(_line((a, c_), (b, d_), layer="grid_d"))
            if ok_seg(a, d_, b, c_, all_w_infl, DISTF=0):
                grid_entities.append(_line((a, d_), (b, c_), layer="grid_d"))

    aeq_entities, uni_entities = [], []
    for (i, j), Aeqv in Aeq_dict.items():
        xL, xR = cols[j].x_axis, cols[j + 1].x_axis
        yB, yT = beams[i].y_axis, beams[i + 1].y_axis
        pid = f"{i},{j}"
        aeq_entities.append(_line((xL, yB), (xR, yT), layer="diag_eq", stroke="#d00", width=2, dash=[6, 4], panel=pid))
        aeq_entities.append(_line((xL, yT), (xR, yB), layer="diag_eq", stroke="#d00", width=2, dash=[6, 4], panel=pid))
        xc, yc = (xL + xR) / 2, (yB + yT) / 2
        aeq_entities.append(_text((xc + 3, yc + 2), f"Aeq={Aeqv:.0f} mm²", fill="#d00", size=10, panel=pid))

        uni_entities.append(_line((xL, yB), (xR, yT), layer="diag_uni", stroke="#0a0", width=2, dash=[6, 4], panel=pid))
        uni_entities.append(_line((xL, yT), (xR, yB), layer="diag_uni", stroke="#0a0", width=2, dash=[6, 4], panel=pid))

    if Yall:
        uni_entities.append(_text((Xall[0], max(Yall) + 15), f"Aunivoca={Aeq_univoca:.0f} mm²", fill="#0a0", size=12))

    overlays = [
        {"id": "grid", "title": "Schema di posa Resisto 5.9", "entities": grid_entities},
        {"id": "aeq_by_panel", "title": "Diagonali equivalenti – Aeq per pannello", "entities": aeq_entities},
        {"id": "aeq_univoca", "title": "Diagonali equivalenti – Aeq univoca", "entities": uni_entities},
    ]

    panels = []
    nt = len(beams) - 1
    np__ = len(cols) - 1
    for i in range(nt):
        for j in range(np__):
            xL, xR = cols[j].x_axis, cols[j + 1].x_axis
            yB, yT = beams[i].y_axis, beams[i + 1].y_axis
            pid = f"{i},{j}"
            panels.append(
                {
                    "id": pid,
                    "i": i,
                    "j": j,
                    "bounds": {"xmin": xL, "xmax": xR, "ymin": yB, "ymax": yT},
                    "center": {"x": (xL + xR) / 2, "y": (yB + yT) / 2},
                    "Aeq_mm2": Aeq_dict.get((i, j), 0.0),
                    "Keq_N_per_mm": Keq_dict.get((i, j), 0.0),
                    "openings": [{"x": w.x, "y": w.y_abs, "w": w.w, "h": w.h} for w in win_data.get((i, j), [])],
                }
            )

    return {
        "job_id": cfg["job_id"],
        "units": "cm",
        "meta": cfg["meta_norm"],
        "geometry": {
            "structure": {
                "beams": [{"y": b.y_axis, "sp": b.spess} for b in beams],
                "columns": [{"x": c.x_axis, "sp": c.spess} for c in cols],
                "windows": [{"x": w.x, "y": w.y_abs, "w": w.w, "h": w.h} for w in all_w_real],
            },
            "panels": panels,
        },
        "results": {
            "Aeq_univoca_mm2": float(round(Aeq_univoca, 2)),
            "Aeq_by_panel_mm2": {f"{i},{j}": v for (i, j), v in Aeq_dict.items()},
            "Keq_by_panel_N_per_mm": {f"{i},{j}": v for (i, j), v in Keq_dict.items()},
            "stats": stats,
            "stats_table": stats_table,
        },
        "overlays": overlays,
        "internals": {
            "Xall": Xall,
            "Yall": Yall,
            "Xbase": Xbase,
            "Ybase": Ybase,
            "matrices_for_pdf": matrices_for_pdf,
            "Aeq_dict": Aeq_dict,
            "Keq_dict": Keq_dict,
            "beams": beams,
            "cols": cols,
            "all_w_real": all_w_real,
            "all_w_infl": all_w_infl,
            "win_data": win_data,
            "DISTF": DISTF,
            "v_segs_pruned": v_segs,
            "h_segs_pruned": h_segs,
        },
    }


# ============================================================
# EXPORTS TO DIRECTORY (TEMP) + ZIP IN RAM
# ============================================================
def render_exports_to_dir(payload: Dict[str, Any], computed: Dict[str, Any], out_dir: Path) -> Dict[str, Optional[Path]]:
    cfg = parse_payload(payload)

    export_png = cfg["export_png"]
    export_pdf = cfg["export_pdf"]
    export_dxf = cfg["export_dxf"]
    if not (export_png or export_pdf or export_dxf):
        export_png = export_pdf = export_dxf = True

    job_id = cfg["job_id"]
    meta_norm = cfg["meta_norm"]

    Xall = computed["internals"]["Xall"]
    Yall = computed["internals"]["Yall"]
    Xbase = computed["internals"]["Xbase"]
    Ybase = computed["internals"]["Ybase"]

    cols = computed["internals"]["cols"]
    beams = computed["internals"]["beams"]

    all_w_real = computed["internals"]["all_w_real"]
    all_w_infl = computed["internals"]["all_w_infl"]

    matrices_for_pdf = computed["internals"]["matrices_for_pdf"]
    Aeq_dict = computed["internals"]["Aeq_dict"]
    Keq_dict = computed["internals"]["Keq_dict"]
    v_segs = computed["internals"].get("v_segs_pruned", [])
    h_segs = computed["internals"].get("h_segs_pruned", [])

    Aeq_univoca = computed["results"]["Aeq_univoca_mm2"]
    stats = computed["results"]["stats"]

    out_dir.mkdir(parents=True, exist_ok=True)

    schema_png = out_dir / f"{job_id}_rinforzo.png"
    grafico1_png = out_dir / f"{job_id}_diag_eq.png"
    grafico2_png = out_dir / f"{job_id}_diag_eq_uni.png"
    dxf_path = out_dir / f"{job_id}_schema_posa_resisto59.dxf"
    first_pdf = out_dir / f"{job_id}_00_schema_statistiche.pdf"
    extra_pdf = out_dir / f"{job_id}_01_extra.pdf"
    final_pdf = out_dir / f"{job_id}_Report_resisto59_completo.pdf"

    header_lines = [
        f"Project: {meta_norm['project_name']}",
        f"Location: {meta_norm['location_name']}",
        f"Wall: {meta_norm['wall_orientation']} | Suffix: {meta_norm['suffix']}",
    ]

    written: Dict[str, Optional[Path]] = {"schema_png": None, "grafico1_png": None, "grafico2_png": None, "pdf_final": None, "dxf": None}

    if export_png or export_pdf:
        def base_axes(ax):
            x_min = cols[0].x_axis - cols[0].spess / 2
            x_max = cols[-1].x_axis + cols[-1].spess / 2
            y_min = beams[0].y_axis - beams[0].spess / 2
            y_max = beams[-1].y_axis + beams[-1].spess / 2

            for b in beams:
                ax.add_patch(plt.Rectangle((x_min, b.y_axis - b.spess / 2), x_max - x_min, b.spess, fc="#d0d0d0", ec="none"))
            for c in cols:
                ax.add_patch(plt.Rectangle((c.x_axis - c.spess / 2, y_min), c.spess, y_max - y_min, fc="#a0a0a0", ec="none"))
            for w in all_w_real:
                ax.add_patch(plt.Rectangle((w.x, w.y_abs), w.w, w.h, fill=False, ec="blue", lw=1.4))

            ax.set_xlim(x_min - 10, x_max + 10)
            ax.set_ylim(y_min - 10, y_max + 10)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel("Y [cm]")
            ax.grid(True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_aspect("equal")
        base_axes(ax)
        ax.set_title("Schema di Posa Resisto 5.9")
        for spine in ax.spines.values():
            spine.set_visible(False)

        for x, y1, y2 in v_segs:
            ax.plot([x, x], [y1, y2], "k", lw=0.7)
        for y, x1, x2 in h_segs:
            ax.plot([x1, x2], [y, y], "k", lw=0.7)

        for i in range(len(Xall) - 1):
            for j in range(len(Yall) - 1):
                a, b = Xall[i], Xall[i + 1]
                c_, d_ = Yall[j], Yall[j + 1]
                if ok_seg(a, c_, b, d_, all_w_infl, DISTF=0):
                    ax.plot([a, b], [c_, d_], "k", lw=0.7)
                if ok_seg(a, d_, b, c_, all_w_infl, DISTF=0):
                    ax.plot([a, b], [d_, c_], "k", lw=0.7)

        fig.tight_layout()
        fig.savefig(schema_png, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_aspect("equal")
        base_axes(ax)
        ax.set_title("Diagonali equivalenti – Aeq per pannello")
        for (i, j), Aeqv in Aeq_dict.items():
            xL, xR = cols[j].x_axis, cols[j + 1].x_axis
            yB, yT = beams[i].y_axis, beams[i + 1].y_axis
            ax.plot([xL, xR], [yB, yT], "r--", lw=1.2)
            ax.plot([xL, xR], [yT, yB], "r--", lw=1.2)
            xc, yc = (xL + xR) / 2, (yB + yT) / 2
            ax.text(xc + 3, yc + 2, f"Aeq={Aeqv:.0f} mm²", fontsize=8, color="red")
        fig.tight_layout()
        fig.savefig(grafico1_png, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_aspect("equal")
        base_axes(ax)
        ax.set_title("Diagonali equivalenti – Aeq Univoca per pannello")
        for (i, j) in Aeq_dict.keys():
            xL, xR = cols[j].x_axis, cols[j + 1].x_axis
            yB, yT = beams[i].y_axis, beams[i + 1].y_axis
            ax.plot([xL, xR], [yB, yT], "g--", lw=1.2)
            ax.plot([xL, xR], [yT, yB], "g--", lw=1.2)
            xc, yc = (xL + xR) / 2, (yB + yT) / 2
            ax.text(xc + 3, yc + 2, f"Aeq={Aeq_univoca:.0f} mm²", fontsize=8, color="green")
        fig.tight_layout()
        fig.savefig(grafico2_png, dpi=300)
        plt.close(fig)

        if export_png:
            written["schema_png"] = schema_png
            written["grafico1_png"] = grafico1_png
            written["grafico2_png"] = grafico2_png

        if export_pdf:
            _first_page(schema_png, stats, first_pdf, header_lines=header_lines)
            _extra_pages(matrices_for_pdf, Aeq_dict, Keq_dict, grafico1_png, grafico2_png, Aeq_univoca, extra_pdf)

            merged = False
            if PdfWriter is not None:
                wr = PdfWriter()
                for p in PdfReader(str(first_pdf)).pages:
                    wr.add_page(p)
                for p in PdfReader(str(extra_pdf)).pages:
                    wr.add_page(p)
                with final_pdf.open("wb") as f:
                    wr.write(f)
                merged = True

            if merged:
                written["pdf_final"] = final_pdf

    if export_dxf:
        dxf_ok = _export_dxf(
            cols, beams,
            finestre_real=all_w_real,
            finestre_infl=all_w_infl,
            X=Xall, Y=Yall,
            Xbase=Xbase, Ybase=Ybase,
            path=dxf_path
        )
        if dxf_ok:
            written["dxf"] = dxf_path

    return written


# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/preview")
def preview(payload: Payload):
    try:
        data = payload.root
        data = {**data, "export": {"png": False, "pdf": False, "dxf": False}}
        computed = compute(data)
        computed.pop("internals", None)
        return {"ok": True, **computed}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/export")
def export(payload: Payload):
    try:
        data = payload.root
        computed = compute(data)
        job_id = computed["job_id"]

        with tempfile.TemporaryDirectory(prefix="rebarca_") as tmp:
            out_dir = Path(tmp)
            _ = render_exports_to_dir(data, computed, out_dir=out_dir)

            mem = BytesIO()
            with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in sorted(out_dir.glob(f"{job_id}_*")):
                    if p.is_file():
                        z.write(p, arcname=p.name)
            mem.seek(0)

            filename = f"{job_id}_allegati.zip"
            return StreamingResponse(
                mem,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
