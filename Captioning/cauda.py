#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import math
import itertools

# ================== CONFIG ==================
CSV_PATH = '/media/cmlre/Extreme Pro/otolith/text_data/filename2features.csv'
DESC_COL = 'Description'   # change if different

# Candidate ID column names (first match is used; else a numeric index is created)
CANDIDATE_ID_COLS = [
    'filename', 'file', 'image', 'img', 'Image', 'ImageName', 'name', 'id', 'ID'
]

# Headings that may FOLLOW the "Cauda" section (to detect its end).
NEXT_HEADINGS = [
    'rostrum', 'antirostrum', 'excisura', 'collum',
    'pars ostialis', 'pars media', 'ostium', 'sulcus acusticus',
    'margin', 'outline', 'shape', 'color', 'colour',
    'dorsal', 'ventral'
]

# Acceptable ways the start heading might be written for Cauda
START_HEADINGS = [
    'cauda',          # canonical
    'pars caudalis',  # synonym in some descriptions
    'caudal region'   # looser phrasing
]

# ------------- circular plot config (optional) -------------
MAKE_CIRCULAR_PLOT = True
TOP_N = 20                     # top-N most frequent Cauda points to show
MAX_ARCS = 250                 # cap arcs to avoid clutter
MIN_CO = None                  # if None -> auto-threshold (>= median or >=2)
OUT_SECTIONS = 'cauda_sections.csv'
OUT_POINTS   = 'cauda_points_exploded.csv'
OUT_ARC_PNG  = 'cauda_points_arc_diagram.png'
OUT_ARC_SVG  = 'cauda_points_arc_diagram.svg'

# ================== HELPERS ==================
def normalize_ws(text: str) -> str:
    """Lowercase and collapse whitespace/hyphens to simplify regex matching."""
    t = text.lower().replace('–', '-').replace('—', '-')
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def _heading_regex(label: str) -> str:
    """Heading tolerant to spaces/hyphens + optional ':' or '-' after heading."""
    return r'\b' + re.escape(label).replace(r'\ ', r'(?:\s+|-)') + r'\b\s*(?::|-)?\s*'

def find_section_bounds(text_norm: str,
                        start_headings: list,
                        next_headings: list):
    """
    Return (start_idx, end_idx, matched_start) of the section that begins at the
    earliest occurrence of any label in start_headings and ends before the next
    heading (from next_headings) or end-of-text.
    """
    starts = []
    for h in start_headings:
        m = re.search(_heading_regex(h), text_norm, flags=re.I)
        if m:
            starts.append((m.start(), m.end(), h))
    if not starts:
        return None
    starts.sort(key=lambda x: x[0])
    start_idx = starts[0][1]
    matched_start = starts[0][2]

    if not next_headings:
        end_idx = len(text_norm)
    else:
        end_union = r'(' + r'|'.join(_heading_regex(h) for h in next_headings) + r')'
        m_end = re.search(end_union, text_norm[start_idx:], flags=re.I)
        end_idx = (start_idx + m_end.start()) if m_end else len(text_norm)

    while end_idx > start_idx and text_norm[end_idx-1] in ' ;,.-':
        end_idx -= 1

    if end_idx <= start_idx:
        return None
    return (start_idx, end_idx, matched_start)

def extract_section_raw(description: str,
                        start_headings=START_HEADINGS,
                        next_headings=NEXT_HEADINGS):
    """Extract raw text of the 'Cauda' section (or synonyms); return None if absent."""
    if not isinstance(description, str) or not description.strip():
        return None
    desc_norm = normalize_ws(description)
    bounds = find_section_bounds(desc_norm, start_headings, next_headings)
    if not bounds:
        return None
    start_idx, end_idx, _ = bounds
    section = desc_norm[start_idx:end_idx].strip(' ;,.-')
    return section if section else None

def safe_split_points(section_text: str):
    """
    Split raw section text into points using ';' and ',' as primary delimiters,
    avoid splitting inside parentheses, normalize 'and' / '&' to commas.
    """
    if not section_text:
        return []
    s = re.sub(r'\s+(and|&)\s+', ', ', section_text)

    parts, current, depth = [], [], 0
    for ch in s:
        if ch == '(':
            depth += 1; current.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1); current.append(ch)
        elif ch in [';', ','] and depth == 0:
            token = ''.join(current).strip(' ;,.-\t\n')
            if token:
                parts.append(token)
            current = []
        else:
            current.append(ch)
    token = ''.join(current).strip(' ;,.-\t\n')
    if token:
        parts.append(token)

    cleaned = []
    for p in parts:
        p = re.sub(r'^(?:type|is|are)\s*[-:]\s*', '', p).strip()
        p = re.sub(r'\s+', ' ', p).strip()
        if p:
            cleaned.append(p)
    return cleaned

def norm_point(s: str) -> str:
    if not isinstance(s, str):
        return ''
    t = s.strip().lower().replace('–', '-').replace('—', '-')
    return ' '.join(t.split())

# ================== LOAD & PROCESS ==================
df = pd.read_csv(CSV_PATH)

if DESC_COL not in df.columns:
    raise ValueError(f"Column '{DESC_COL}' not found in {CSV_PATH}. "
                     f"Available: {list(df.columns)}")

id_col = next((c for c in CANDIDATE_ID_COLS if c in df.columns), None)
if id_col is None:
    df['_row'] = np.arange(len(df))
    id_col = '_row'

# 1) Extract raw "Cauda" section
df['Cauda_text'] = df[DESC_COL].apply(extract_section_raw)

# 2) Split into individual points (list[str])
df['Cauda_points'] = df['Cauda_text'].apply(safe_split_points)

# 3) Explode to one point per row
exploded = (
    df[[id_col, 'Cauda_text', 'Cauda_points']]
    .explode('Cauda_points', ignore_index=True)
    .rename(columns={
        id_col: 'RecordID',
        'Cauda_text': 'Cauda_section',
        'Cauda_points': 'Cauda_point'
    })
)
exploded = exploded.dropna(subset=['Cauda_point'])
exploded = exploded[exploded['Cauda_point'].str.len() > 0]
exploded['point_norm'] = exploded['Cauda_point'].apply(norm_point)

# ================== SAVE ==================
df[[id_col, 'Cauda_text', 'Cauda_points']].to_csv(OUT_SECTIONS, index=False)
exploded[['RecordID','Cauda_section','Cauda_point','point_norm']].to_csv(OUT_POINTS, index=False)

print(f"Extracted Cauda section for {df['Cauda_text'].notna().sum()} / {len(df)} records.")
print(f"Saved sections → {OUT_SECTIONS}")
print(f"Saved exploded points → {OUT_POINTS}")

# ================== OPTIONAL: CIRCULAR CO-OCCURRENCE PLOT ==================
if MAKE_CIRCULAR_PLOT and not exploded.empty:
    freq = exploded['point_norm'].value_counts()
    top_points = list(freq.head(TOP_N).index)

    grouped = exploded[exploded['point_norm'].isin(top_points)].groupby('RecordID')['point_norm'].apply(lambda s: sorted(set(s)))
    pairs = []
    for _, pts in grouped.items():
        for a, b in itertools.combinations(pts, 2):
            pairs.append((a, b))
            pairs.append((b, a))  # symmetric

    co_counts = pd.Series(pairs).value_counts() if len(pairs) else pd.Series(dtype=int)

    # Create symmetric co-occurrence matrix among top points
    idx = pd.Index(top_points, name='point')
    co_mat = pd.DataFrame(0, index=idx, columns=idx, dtype=int)
    for (a, b), c in co_counts.items():
        if a in co_mat.index and b in co_mat.columns:
            co_mat.loc[a, b] = c
    for p in top_points:
        co_mat.loc[p, p] = freq[p]

    # Build edges (only off-diagonal)
    edges = []
    for i, a in enumerate(top_points):
        for j, b in enumerate(top_points):
            if j <= i: continue
            w = co_mat.loc[a, b]
            if w > 0:
                edges.append((a, b, int(w)))

    if edges:
        weights = np.array([w for _, _, w in edges])
        if MIN_CO is None:
            MIN_CO = max(2, int(np.percentile(weights, 50)))  # ≥ median or 2
        edges = [e for e in edges if e[2] >= MIN_CO]
        edges = sorted(edges, key=lambda x: x[2], reverse=True)[:MAX_ARCS]

    # Layout on a circle
    n = len(top_points)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    positions = {p: (math.cos(a), math.sin(a)) for p, a in zip(top_points, angles)}

    def draw_arc(ax, p1, p2, weight, wmin, wmax):
        (x1, y1) = positions[p1]; (x2, y2) = positions[p2]
        verts = [(x1, y1), (0.0, 0.0), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        if wmax == wmin:
            lw, alpha = 1.5, 0.6
        else:
            t = (weight - wmin) / (wmax - wmin)
            lw = 0.5 + 4.0 * t
            alpha = 0.3 + 0.5 * t
        ax.add_patch(PathPatch(path, fill=False, linewidth=lw, alpha=alpha))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'Cauda: Arc Diagram of Co-occurrence (Top-{len(top_points)} Points)')

    # nodes + labels
    for p in top_points:
        x, y = positions[p]
        ax.plot([x], [y], marker='o', markersize=5)
        ang = math.atan2(y, x); lx, ly = 1.15*math.cos(ang), 1.15*math.sin(ang)
        rot = math.degrees(ang)
        ax.text(lx, ly, p.title(), ha='center', va='center', rotation=rot if -90 <= rot <= 90 else rot+180)

    if edges:
        wmin = min(w for _, _, w in edges); wmax = max(w for _, _, w in edges)
        for a, b, w in edges:
            draw_arc(ax, a, b, w, wmin, wmax)

    fig.tight_layout()
    fig.savefig(OUT_ARC_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_ARC_SVG, bbox_inches='tight')
    print(f"Saved circular plot → {OUT_ARC_PNG} and {OUT_ARC_SVG}")
