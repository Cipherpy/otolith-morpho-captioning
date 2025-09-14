#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Radial bars for 4 CSVs using the *entire cell text* as labels,
but drawn in a decluttered way:
 - Keep only TOP_N most frequent full-text entries per region
 - Group the rest into 'Other'
 - Wrap/shorten labels and add small rotation to avoid overlap
 - Transparent, publication-ready PNG

Columns required (one per CSV):
  SulcusAcusticus_text, Cauda_text, Ostium_text, Posterior_text
"""

import math
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
CSV_SULCUS    = "/home/cmlre/Desktop/otolith/sulcus_acusticus_sections.csv"
CSV_CAUDA     = "/home/cmlre/Desktop/otolith/cauda_sections.csv"
CSV_OSTIUM    = "/home/cmlre/Desktop/otolith/ostium_sections.csv"
CSV_POSTERIOR = "/home/cmlre/Desktop/otolith/posterior_sections.csv"

TEXT_COLS = {
    "Sulcus acusticus": "SulcusAcusticus_text",
    "Cauda":            "Cauda_text",
    "Ostium":           "Ostium_text",
    "Posterior":        "Posterior_text",
}

# De-cluttering knobs
TOP_N            = 6      # keep only top N full-text entries per region
MIN_PCT_TO_KEEP  = 3.0    # drop items below this % unless in top N
WRAP_WIDTH       = 24     # wrap label text to this many characters
MAX_LINES        = 2      # max lines for wrapped labels (then ellipsis)
GLOBAL_MAX       = 50.0   # radial limit (percent axis)
BAR_GAP_DEG      = 3      # gap between neighboring bars (degrees)
SECTOR_GAP_DEG   = 12     # gap between region sectors (degrees)
START_ANGLE_DEG  = 90     # start from top

OUT_PNG = "otolith_radial_fulltext_decluttered.png"

# Aesthetic: compact sans-serif like Nature figures
mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Colors per region (distinct, slightly muted)
REGION_COLORS = {
    "Sulcus acusticus": "#1f77b4",
    "Cauda":            "#ff7f0e",
    "Ostium":           "#2ca02c",
    "Posterior":        "#9467bd",
}

# ------------- helpers -------------
def wrap_label(s: str, width=WRAP_WIDTH, max_lines=MAX_LINES) -> str:
    return str(s).strip()

def region_counts(csv_path, col_name, region_name):
    df = pd.read_csv(csv_path)
    if col_name not in df.columns:
        raise ValueError(f"{region_name}: column '{col_name}' not found in {csv_path}")
    vals = df[col_name].dropna().astype(str).tolist()
    total = max(1, len(vals))
    cnt = Counter(vals)
    # build (label, pct) pairs
    pairs = [(k, 100.0*v/total) for k, v in cnt.items()]
    # sort by pct desc, then label
    pairs.sort(key=lambda x: (-x[1], x[0]))

    # keep top-N; also keep items >= MIN_PCT_TO_KEEP; group rest as "Other"
    kept = pairs[:TOP_N]
    others = [(k, p) for (k, p) in pairs[TOP_N:] if p >= MIN_PCT_TO_KEEP]
    discarded = [(k, p) for (k, p) in pairs if (k, p) not in kept and (k, p) not in others]
    other_pct = sum(p for _, p in discarded)

    labels = [k for k, _ in kept + others]
    values = [p for _, p in kept + others]
    if other_pct > 0:
        labels.append(f"Other ({len(discarded)})")
        values.append(other_pct)
    return labels, values

# ------------- load all regions -------------
regions = [
    ("Sulcus acusticus", CSV_SULCUS),
    ("Cauda",            CSV_CAUDA),
    ("Ostium",           CSV_OSTIUM),
    ("Posterior",        CSV_POSTERIOR),
]

region_info = []
for name, path in regions:
    labs, vals = region_counts(path, TEXT_COLS[name], name)
    # clip to GLOBAL_MAX (values are %)
    vals = [min(v, GLOBAL_MAX) for v in vals]
    region_info.append({"region": name, "labels": labs, "values": vals})

# ------------- plotting -------------
fig = plt.figure(figsize=(9.8, 9.8))
# Transparent figure background
fig.patch.set_alpha(0.0)

ax = plt.subplot(111, polar=True)
# Transparent axes background
ax.set_facecolor("none")
ax.set_ylim(0, GLOBAL_MAX)

# radial guide rings
for r in np.arange(10, GLOBAL_MAX + 0.1, 10):
    ax.plot(np.linspace(0, 2*np.pi, 361), [r]*361, color="#E5E5E5", lw=0.7, zorder=0)

ax.set_yticks([10, 20, 30, 40, 50])
ax.set_yticklabels([str(x) for x in [10,20,30,40,50]])
ax.set_xticks([])

# sector sizes proportional to item count (still spaced out by sector gaps)
total_slots = sum(len(r["labels"]) for r in region_info)
sector_gap = math.radians(SECTOR_GAP_DEG)
bar_gap = math.radians(BAR_GAP_DEG)

angle_full = 2*np.pi - len(region_info)*sector_gap
angle_per_slot = angle_full / max(1, total_slots)

theta = math.radians(START_ANGLE_DEG)
sectors = []
for r in region_info:
    width = len(r["labels"]) * angle_per_slot
    sectors.append((theta, theta + width))
    theta += width + sector_gap

# draw bars + labels
for (theta0, theta1), reg in zip(sectors, region_info):
    region = reg["region"]
    color = REGION_COLORS.get(region, "#444")
    n = len(reg["labels"])
    if n == 0:
        continue
    slot = (theta1 - theta0) / n
    # region title
    mid = (theta0 + theta1) / 2
    ax.text(mid, GLOBAL_MAX + 5.5, region,
            rotation=np.degrees(mid - np.pi/2),
            rotation_mode="anchor", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#111")

    for i, (lab, val) in enumerate(zip(reg["labels"], reg["values"])):
        left = theta0 + i*slot + bar_gap/2
        width = max(slot - bar_gap, math.radians(1.5))

        # bar
        ax.bar(left, val, width=width, bottom=0,
               color=color, alpha=0.9, edgecolor="white", linewidth=0.6, align="edge")

        # label placement (NO extra padding — sit right at the bar tip)
        angle_mid = left + width/2
        deg = np.degrees(angle_mid)
        r_label = min(GLOBAL_MAX - 0.5, val + 0.5)  # no extra outward padding

        nice_label = wrap_label(lab, width=WRAP_WIDTH, max_lines=MAX_LINES)

        # Make left-side labels readable by flipping 180° and right-aligning
        if 90 < deg < 270:
            rot = deg - 180
            ax.text(angle_mid, r_label, nice_label,
                    rotation=rot, rotation_mode="anchor",
                    ha="right", va="center", fontsize=4, color=color)
        else:
            rot = deg
            ax.text(angle_mid, r_label, nice_label,
                    rotation=rot, rotation_mode="anchor",
                    ha="left", va="center", fontsize=4, color=color)

# legend
legend_handles = [
    mpl.patches.Patch(facecolor=REGION_COLORS[r["region"]], edgecolor="white", label=r["region"])
    for r in region_info
]
ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
          ncol=4, frameon=False, title="% of records (top items; others grouped)")

ax.grid(False)
plt.tight_layout(pad=2.0)
plt.savefig(
    OUT_PNG,
    dpi=1500,           # high resolution
    transparent=True,  # transparent background
    bbox_inches="tight",  # trim whitespace
    pad_inches=0.2       # small padding around
)
print(f"Saved: {OUT_PNG}")
