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

# Other section headings that may follow "Sulcus acusticus"
# (extend this list to fit your schema/wording)
NEXT_HEADINGS = [
    'ostium', 'cauda', 'rostrum', 'antirostrum', 'excisura', 'collum',
    'pars ostialis', 'pars caudalis', 'pars media',
    'dorsal', 'ventral', 'margin', 'outline', 'shape', 'color', 'colour',
    'sulcus acusticus'  # if mentioned again later
]

# ================== HELPERS ==================
def normalize_ws(text: str) -> str:
    """Lowercase and collapse whitespace/hyphens to simplify regex matching."""
    # keep original for output; operate on a normalized copy
    t = text.lower()
    # unify hyphen-like dashes
    t = t.replace('–', '-').replace('—', '-')
    # collapse whitespace
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def find_section_bounds(text_norm: str, start_heading: str, next_headings: list):
    """
    Return (start_idx, end_idx) of the section that begins at 'start_heading'
    and ends right before the next heading or end-of-string. Indices refer to
    positions in the original normalized string.
    """
    # Build a heading matcher tolerant to hyphen or space between words
    # e.g., "sulcus acusticus", "sulcus-acusticus"
    heading_pattern = r'\b' + re.escape(start_heading).replace(r'\ ', r'(?:\s+|-)') + r'\b'
    start_match = re.search(heading_pattern + r'\s*(?::|-)?\s*', text_norm, flags=re.I)
    if not start_match:
        return None

    start_idx = start_match.end()

    if not next_headings:
        return (start_idx, len(text_norm))

    # Construct next-heading regex (same tolerance for spaces/hyphens + optional ":" or "-")
    def heading_to_regex(h):
        return r'\b' + re.escape(h).replace(r'\ ', r'(?:\s+|-)') + r'\b\s*(?::|-)?\s*'

    end_regex = r'(' + r'|'.join(heading_to_regex(h) for h in next_headings) + r')'
    m_end = re.search(end_regex, text_norm[start_idx:], flags=re.I)

    if m_end:
        end_idx = start_idx + m_end.start()
    else:
        end_idx = len(text_norm)

    # Trim trailing separators
    while end_idx > start_idx and text_norm[end_idx-1] in ' ;,.-':
        end_idx -= 1

    if end_idx <= start_idx:
        return None
    return (start_idx, end_idx)

def extract_section_raw(description: str,
                        start_heading='sulcus acusticus',
                        next_headings=NEXT_HEADINGS):
    """Extract raw text of the 'Sulcus acusticus' section; return None if absent."""
    if not isinstance(description, str) or not description.strip():
        return None
    desc_norm = normalize_ws(description)
    bounds = find_section_bounds(desc_norm, start_heading, next_headings)
    if not bounds:
        return None
    start_idx, end_idx = bounds
    section = desc_norm[start_idx:end_idx].strip(' ;,.-')
    return section if section else None

_SPLIT_GUARDS = re.compile(r'\([^)]*\)')  # to avoid splitting inside parentheses

def safe_split_points(section_text: str):
    """
    Split the raw section text into 'points' using ';' and ',' as primary delimiters,
    but avoid splitting inside parentheses. Also normalizes ' and ' / ' & ' to commas.
    """
    if not section_text:
        return []

    # Normalize common conjunctions to comma
    s = re.sub(r'\s+(and|&)\s+', ', ', section_text)

    # We only split on commas/semicolons that are NOT inside parentheses
    parts = []
    current = []
    depth = 0  # parenthesis nesting depth

    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch in [';', ','] and depth == 0:
            token = ''.join(current).strip(' ;,.-\t\n')
            if token:
                parts.append(token)
            current = []
        else:
            current.append(ch)

    # Last token
    token = ''.join(current).strip(' ;,.-\t\n')
    if token:
        parts.append(token)

    # Clean up each point
    cleaned = []
    for p in parts:
        # drop leading markers like "type -", "is -"
        p = re.sub(r'^(?:type|is|are)\s*[-:]\s*', '', p).strip()
        # collapse leftover whitespace
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

# 1) Extract the raw "Sulcus acusticus" section
df['SulcusAcusticus_text'] = df[DESC_COL].apply(extract_section_raw)

# 2) Split into individual points (list[str])
df['SulcusAcusticus_points'] = df['SulcusAcusticus_text'].apply(safe_split_points)

# 3) Explode to one point per row for easy analysis
exploded = (
    df[[id_col, 'SulcusAcusticus_text', 'SulcusAcusticus_points']]
    .explode('SulcusAcusticus_points', ignore_index=True)
    .rename(columns={
        id_col: 'RecordID',
        'SulcusAcusticus_text': 'SulcusAcusticus_section',
        'SulcusAcusticus_points': 'SulcusAcusticus_point'
    })
)

# Drop rows where the section/point is missing
exploded = exploded.dropna(subset=['SulcusAcusticus_point'])
exploded = exploded[exploded['SulcusAcusticus_point'].str.len() > 0]

# ================== SAVE ==================
OUT_SECTIONS = 'sulcus_acusticus_sections.csv'
OUT_POINTS   = 'sulcus_acusticus_points_exploded.csv'

df[['{}{}'.format('', id_col if id_col != '_row' else 'index'),
    'SulcusAcusticus_text', 'SulcusAcusticus_points']].to_csv(OUT_SECTIONS, index=False)

exploded.to_csv(OUT_POINTS, index=False)

# ================== REPORT ==================
print(f"Extracted Sulcus acusticus section for {df['SulcusAcusticus_text'].notna().sum()} / {len(df)} records.")
print("Example (first 5 exploded points):")
print(exploded.head(5).to_string(index=False))
print(f"\nSaved:\n- Sections per record → {OUT_SECTIONS}\n- One-point-per-row → {OUT_POINTS}")
