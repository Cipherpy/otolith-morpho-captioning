#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd

# ================== CONFIG ==================
CSV_PATH = '/media/cmlre/Extreme Pro/otolith/text_data/filename2features.csv'
DESC_COL = 'Description'   # change if your column is named differently

# If present, one of these will be used as an ID column in outputs
CANDIDATE_ID_COLS = [
    'filename', 'file', 'image', 'img', 'Image', 'ImageName', 'name', 'id', 'ID'
]

# Section names that may FOLLOW the “Ostium” section (to detect its end)
NEXT_HEADINGS = [
    'cauda', 'sulcus acusticus', 'rostrum', 'antirostrum', 'excisura', 'collum',
    'pars caudalis', 'pars media', 'margin', 'outline', 'shape', 'color', 'colour',
    'dorsal', 'ventral'
]

# Acceptable ways the start heading might be written
START_HEADINGS = [
    'ostium',          # canonical
    'pars ostialis',   # sometimes used synonymously
    'ostial region'    # looser phrasing seen in some notes
]

# ================== HELPERS ==================
def normalize_ws(text: str) -> str:
    """Lowercase and collapse whitespace/hyphens to simplify regex matching."""
    t = text.lower()
    t = t.replace('–', '-').replace('—', '-')
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def _heading_regex(label: str) -> str:
    """
    Heading tolerant to spaces/hyphens + optional ':' or '-' after heading.
    e.g., 'ostium', 'pars ostialis' → matches 'pars-ostialis:'
    """
    return r'\b' + re.escape(label).replace(r'\ ', r'(?:\s+|-)') + r'\b\s*(?::|-)?\s*'

def find_section_bounds(text_norm: str,
                        start_headings: list,
                        next_headings: list):
    """
    Return (start_idx, end_idx, matched_start) of the section that begins at the earliest
    occurrence of any label in start_headings and ends right before the next heading or EoT.
    Indices refer to positions in the normalized string.
    """
    # Find the earliest start among provided headings
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

    # If no "next" headings are provided, capture to end
    if not next_headings:
        end_idx = len(text_norm)
    else:
        # Build union regex for all possible next headings
        end_union = r'(' + r'|'.join(_heading_regex(h) for h in next_headings) + r')'
        m_end = re.search(end_union, text_norm[start_idx:], flags=re.I)
        end_idx = (start_idx + m_end.start()) if m_end else len(text_norm)

    # Trim trailing separators
    while end_idx > start_idx and text_norm[end_idx-1] in ' ;,.-':
        end_idx -= 1

    if end_idx <= start_idx:
        return None
    return (start_idx, end_idx, matched_start)

def extract_section_raw(description: str,
                        start_headings=START_HEADINGS,
                        next_headings=NEXT_HEADINGS):
    """Extract raw text of the 'Ostium' section (or synonyms); return None if absent."""
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
    Split the raw section text into 'points' using ';' and ',' as primary delimiters,
    avoid splitting inside parentheses, and normalize 'and' / '&' to commas.
    """
    if not section_text:
        return []

    # Normalize common conjunctions to comma
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

# ================== LOAD & PROCESS ==================
df = pd.read_csv(CSV_PATH)

if DESC_COL not in df.columns:
    raise ValueError(f"Column '{DESC_COL}' not found in {CSV_PATH}. "
                     f"Available columns: {list(df.columns)}")

# Pick an ID column if one exists; else use a numeric index
id_col = next((c for c in CANDIDATE_ID_COLS if c in df.columns), None)
if id_col is None:
    df['_row'] = np.arange(len(df))
    id_col = '_row'

# 1) Extract the raw "Ostium" section
df['Ostium_text'] = df[DESC_COL].apply(extract_section_raw)

# 2) Split into individual points (list[str])
df['Ostium_points'] = df['Ostium_text'].apply(safe_split_points)

# 3) Explode to one point per row for easy analysis
exploded = (
    df[[id_col, 'Ostium_text', 'Ostium_points']]
    .explode('Ostium_points', ignore_index=True)
    .rename(columns={
        id_col: 'RecordID',
        'Ostium_text': 'Ostium_section',
        'Ostium_points': 'Ostium_point'
    })
)

# Drop rows where the section/point is missing
exploded = exploded.dropna(subset=['Ostium_point'])
exploded = exploded[exploded['Ostium_point'].str.len() > 0]

# ================== SAVE ==================
OUT_SECTIONS = 'ostium_sections.csv'
OUT_POINTS   = 'ostium_points_exploded.csv'

# Save sections (+ list of points per record)
df[[id_col, 'Ostium_text', 'Ostium_points']].to_csv(OUT_SECTIONS, index=False)

# Save exploded one-point-per-row
exploded.to_csv(OUT_POINTS, index=False)

# ================== REPORT ==================
print(f"Extracted Ostium section for {df['Ostium_text'].notna().sum()} / {len(df)} records.")
print("Example (first 5 exploded points):")
print(exploded.head(5).to_string(index=False))
print(f"\nSaved:\n- Sections per record → {OUT_SECTIONS}\n- One-point-per-row → {OUT_POINTS}")
