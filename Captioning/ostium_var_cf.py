#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dot-matrix Confusion Plot (Row-normalized)
- Green = correct (diagonal), Red = wrong (off-diagonal)
- Bubble size ∝ row-normalized value (0–1)
- Transparent, high-resolution PNG output + CSV of normalized CM
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ========================= STYLE =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10,        # base font size (between 5–7 pt)
    "axes.labelsize": 10,   # axis labels
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.title_fontsize": 10,
    "legend.fontsize": 10,
})

# ========================= CONFIG =========================
CSV_IN   = "/home/cmlre/Desktop/otolith_new/paired_captions_minimal copy.csv"
COL_GT   = "Description"
COL_PRED = "generated_caption"

OUT_DIR        = "plots/"
OUT_NORM_CSV   = os.path.join(OUT_DIR, "ostium_cm_normalized.csv")
OUT_PNG        = os.path.join(OUT_DIR, "ostium_dotmatrix_correct_vs_wrong_gemma.png")

# Plot sizing & styles
DOT_MIN   = 8       # min marker area (pt^2)
DOT_MAX   = 180     # max marker area (pt^2)
FIG_DX    = 0.40    # width scale per column
FIG_DY    = 0.35    # height scale per row
LABEL_PAD = 6       # axis label padding
GRID_ALPHA = 0.06
SKIP_BELOW = 1e-6   # do not draw bubbles for (near-)zero values

COLOR_CORRECT = "#1f9d55"  # green
COLOR_WRONG   = "#d64545"  # red
GRID_COLOR    = (0, 0, 0, GRID_ALPHA)

# ===================== HELPERS ============================
def extract_raw_ostium(text: str) -> str:
    """Extract exact raw text after 'Ostium:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r"Ostium:\s*([^\.]*)", str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalize a count matrix to [0,1]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0)
    return norm.fillna(0.0)

# ===================== LOAD & PREP ========================
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_IN)

# Parse GT and predicted raw phrases
df["ostium_gt_raw"]  = df[COL_GT].apply(extract_raw_ostium)
df["ostium_gen_raw"] = df[COL_PRED].apply(extract_raw_ostium)

filtered = df[
    (df["ostium_gt_raw"]  != "") &
    (df["ostium_gen_raw"] != "")
].copy()

# Crosstab (counts)
cm_counts = pd.crosstab(
    filtered["ostium_gt_raw"],
    filtered["ostium_gen_raw"],
    dropna=False
)

# Order rows/cols by total counts for readability
row_order = cm_counts.sum(axis=1).sort_values(ascending=False).index.tolist()
col_order = cm_counts.sum(axis=0).sort_values(ascending=False).index.tolist()
cm_counts = cm_counts.loc[row_order, col_order]

# Row-normalized matrix
cm_norm = row_normalize(cm_counts)
cm_norm.to_csv(OUT_NORM_CSV, index=True)

rows = cm_norm.index.tolist()
cols = cm_norm.columns.tolist()
vals = cm_norm.values

# ===================== PLOT ===============================
fig_w = max(10, FIG_DX * len(cols))
fig_h = max(8,  FIG_DY * len(rows))
plt.figure(figsize=(fig_w, fig_h), dpi=1500)
ax = plt.gca()
ax.set_facecolor("white")

# faint grid
for x in range(len(cols) + 1):
    ax.axvline(x - 0.5, color=GRID_COLOR, lw=0.6, zorder=0)
for y in range(len(rows) + 1):
    ax.axhline(y - 0.5, color=GRID_COLOR, lw=0.6, zorder=0)

# Scatter each (non-zero) cell
ys, xs = np.indices(vals.shape)
xs = xs.ravel()
ys = ys.ravel()
v  = vals.ravel()

# sizes ∝ value
sizes = DOT_MIN + (DOT_MAX - DOT_MIN) * v

# colors by correct vs wrong
plot_x, plot_y, plot_sizes, plot_colors = [], [], [], []
for xi, yi, vi, si in zip(xs, ys, v, sizes):
    if vi <= SKIP_BELOW:
        continue
    rlab, clab = rows[yi], cols[xi]
    is_diag = (rlab == clab)
    plot_x.append(xi)
    plot_y.append(yi)
    plot_sizes.append(si)
    plot_colors.append(COLOR_CORRECT if is_diag else COLOR_WRONG)

ax.scatter(
    plot_x, plot_y,
    s=plot_sizes,
    c=plot_colors,
    marker='o',
    linewidths=0,
    alpha=0.9
)

# ticks & labels
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(rows)))
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticklabels(rows)

# invert y so first row is at top
ax.invert_yaxis()

# spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_alpha(0.3)

# # Legends
# legend_levels = [0.25, 0.50, 0.75, 1.00]
# size_handles = [
#     Line2D([0],[0], marker='o', linestyle='',
#            markersize=np.sqrt(DOT_MIN + (DOT_MAX - DOT_MIN) * lv),
#            color="black", alpha=0.85, label=f"{lv:.2f}")
#     for lv in legend_levels
# ]
# leg1 = ax.legend(
#     handles=size_handles,
#     title="Row-normalized value",
#     frameon=False,
#     loc="lower left",
#     bbox_to_anchor=(-0.55, -0.3),
#     handleheight=2.0,   # increase vertical spacing
#     handlelength=1.5,   # length of legend handle
#     labelspacing=1.2    # spacing between labels
# )
# ax.add_artist(leg1)

# color_handles = [
#     Line2D([0],[0], marker='o', linestyle='', color=COLOR_CORRECT, markersize=6, label="Correct"),
#     Line2D([0],[0], marker='o', linestyle='', color=COLOR_WRONG,   markersize=6, label="Wrong"),
# ]
# ax.legend(
#     handles=color_handles,
#     title="Prediction type",
#     frameon=False,
#     loc="lower left",
#     bbox_to_anchor=(-0.55, -0.05)
# )

ax.set_xlabel("Generated features", labelpad=LABEL_PAD)
ax.set_ylabel("Reference features", labelpad=LABEL_PAD)
ax.set_xlim(-0.5, len(cols)-0.5)
ax.set_ylim(len(rows)-0.5, -0.5)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=1500, transparent=True, bbox_inches="tight")
plt.close()

print(f"Saved:\n- Normalized CM: {OUT_NORM_CSV}\n- Figure: {OUT_PNG}")
