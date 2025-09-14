#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -------------------- STYLE --------------------
mpl.rcParams.update({
    "figure.dpi": 1500,
    "savefig.dpi": 1500,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# -------------------- INPUTS -------------------
GEMMA_CSV = "/home/cmlre/Desktop/Captioning/GRayscale_otolith_paired_captions_scores.csv"
LLAMA_CSV = "/home/cmlre/Desktop/Captioning/llama/Grayscale_llama_otolith_paired_captions_scores.csv"
LABEL_COL = "actual_label"
METRICS   = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L']

# Limit to top-K species (by mean of the two models) to reduce clutter. None = all.
TOP_K = None  # e.g., 30

# Sort key: "mean" or "delta"
SORT_BY = "mean"  # or "delta"

# Colors
COLOR_GEMMA = "#5B5BD6"   # purple-ish (Gemma-3)
COLOR_LLAMA = "#FF7F0E"   # orange (LLaMA 3.2)
CMAP_DELTA  = "coolwarm"  # diverging for Δ

# -------------------- HELPERS -------------------
def prepare_scores(csv_path, label_col=LABEL_COL, metrics=METRICS):
    df = pd.read_csv(csv_path)

    # keep only available metrics
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError(f"No expected metric columns found in {csv_path}")

    # numeric
    df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')
    df['auto_avg'] = df[metrics].mean(axis=1)

    # normalize 0–1 within this file to make scales comparable
    maxv = df['auto_avg'].max()
    df['auto_avg_norm'] = df['auto_avg'] / (maxv if (pd.notnull(maxv) and maxv > 0) else 1.0)

    out = df.groupby(label_col, as_index=False)['auto_avg_norm'].mean()
    return out

# -------------------- LOAD & MERGE -------------------
gemma = prepare_scores(GEMMA_CSV).rename(columns={'auto_avg_norm': 'Gemma-3'})
llama = prepare_scores(LLAMA_CSV).rename(columns={'auto_avg_norm': 'LLaMA 3.2'})

merged = pd.merge(gemma, llama, on=LABEL_COL, how='inner')
if merged.empty:
    raise ValueError("No overlapping classes between the two CSVs. If desired, use how='outer' and fillna(0).")

merged['mean']  = merged[['Gemma-3', 'LLaMA 3.2']].mean(axis=1)
merged['delta'] = merged['Gemma-3'] - merged['LLaMA 3.2']

if SORT_BY == "delta":
    merged = merged.sort_values('delta', ascending=True)  # small to large Δ
else:
    merged = merged.sort_values('mean', ascending=False)  # high to low mean

if TOP_K is not None:
    merged = merged.head(TOP_K).copy()

# Safety bounds for axes
merged['Gemma-3']  = merged['Gemma-3'].clip(0, 1)
merged['LLaMA 3.2'] = merged['LLaMA 3.2'].clip(0, 1)

# -------------------- PLOT -------------------
n = len(merged)
if n == 0:
    raise ValueError("No rows to plot after filtering.")

# dynamic figure size: height scales with number of categories
fig_h = max(6.5, 0.28 * n + 2.0)
fig_w = 12.5 if n < 30 else 13.5

fig, (ax_left, ax_right) = plt.subplots(
    nrows=1, ncols=2, sharey=True,
    figsize=(fig_w, fig_h),
    gridspec_kw={"width_ratios": [3.0, 1.2]}
)

ypos = np.arange(n)

# Build a diverging normalization centered at 0 for Δ colors
abs_max_delta = np.nanmax(np.abs(merged['delta'].values)) if n > 0 else 1.0
abs_max_delta = abs_max_delta if abs_max_delta > 0 else 1.0
norm = TwoSlopeNorm(vmin=-abs_max_delta, vcenter=0.0, vmax=abs_max_delta)
cmap = plt.get_cmap(CMAP_DELTA)

# ---------- LEFT: Dumbbell ----------
# Background grid
ax_left.set_axisbelow(True)
ax_left.grid(axis='x', linestyle=':', alpha=0.5)

# Draw a line per species (color by Δ so it matches the right panel)
for i, (_, row) in enumerate(merged.iterrows()):
    g = float(row['Gemma-3'])
    l = float(row['LLaMA 3.2'])
    d = float(row['delta'])
    line_color = cmap(norm(d))
    ax_left.hlines(y=i, xmin=min(g, l), xmax=max(g, l), color=line_color, lw=2, alpha=0.9)

# Endpoints
ax_left.scatter(merged['LLaMA 3.2'].values, ypos, s=28, color=COLOR_LLAMA, edgecolor='white', linewidths=0.6, zorder=3, label="LLaMA 3.2")
ax_left.scatter(merged['Gemma-3'].values,  ypos, s=28, color=COLOR_GEMMA, edgecolor='white', linewidths=0.6, zorder=3, label="Gemma-3")

# Axes setup
ax_left.set_xlim(-0.02, 1.02)
ax_left.set_xlabel("Normalized score (0–1)", labelpad=6)
ax_left.set_yticks(ypos)
ax_left.set_yticklabels(merged[LABEL_COL].values, ha='right')
ax_left.invert_yaxis()  # top to bottom = best to worst (when sorted by mean)
#ax_left.set_title("Dumbbell: Gemma-3 vs LLaMA 3.2", pad=8)
ax_left.legend(loc="lower right", frameon=False, fontsize=10, ncols=2, handlelength=1.2, columnspacing=1.5)

# ---------- RIGHT: Δ (Gemma − LLaMA) ----------
colors = cmap(norm(merged['delta'].values))
ax_right.barh(ypos, merged['delta'].values, color=colors, edgecolor='none', height=0.6)
ax_right.axvline(0, color='black', lw=0.8, alpha=0.8)

# Expand x-limits symmetrically
pad = 0.05 * abs_max_delta
ax_right.set_xlim(-abs_max_delta - pad, abs_max_delta + pad)
ax_right.set_xlabel("Δ  (Gemma − LLaMA)")
ax_right.set_title("Difference (Δ)", pad=8)

# Optional value labels on the Δ bars (toggle if needed)
SHOW_DELTA_LABELS = False
if SHOW_DELTA_LABELS:
    for i, d in enumerate(merged['delta'].values):
        x = d + (0.01 if d >= 0 else -0.01)
        ha = 'left' if d >= 0 else 'right'
        ax_right.text(x, i, f"{d:+.2f}", va='center', ha=ha, fontsize=10)

# Colormap colorbar for Δ
cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax_right, orientation='vertical', pad=0.08, fraction=0.08
)
cb.set_label("Δ magnitude (centered at 0)")

# Clean up spines
for ax in (ax_left, ax_right):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Tidy layout & save
#fig.suptitle("Gemma-3 vs LLaMA 3.2 — Dumbbell + Δ Plot", y=0.995, fontsize=10)
fig.tight_layout()

os.makedirs("plots", exist_ok=True)
out_png = "plots/grayscale_gemma_vs_llama_dumbbell_delta.png"
out_pdf = "plots/grayscale_gemma_vs_llama_dumbbell_delta.pdf"
plt.savefig(out_png, bbox_inches='tight', dpi=1500,transparent=True)
plt.savefig(out_pdf, bbox_inches='tight', dpi=1500,transparent=True)
print(f"Saved:\n  {os.path.abspath(out_png)}\n  {os.path.abspath(out_pdf)}")

plt.show()
