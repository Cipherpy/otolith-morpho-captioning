#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

# ========================= STYLE =========================
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 1500,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ========================= INPUTS ========================
CSV_PATH = "/home/cmlre/Desktop/Captioning/paired_captions_minimal.csv"  # change if needed
COL_TRUE = "actual_label"
COL_PRED = "predicted_label"

OUT_HM_PNG     = "plots/gemma_confusion_fit_to_screen.png"      # heatmap
OUT_CHORD_PNG  = "plots/gemma_confusion_chord_connect_only.png" # chord: ticks+labels, NO big bars
OUT_ALLUVIAL   = "plots/gemma_confusion_alluvial.png"           # alluvial ribbons

# Connectivity controls
MIN_FLOW_NORM = 0.0      # keep ALL connections (draw anything > 0)
TOP_K_PER_ROW = None     # keep all; set e.g. 4 to limit clutter

# ========================= LOAD ==========================
df = pd.read_csv(CSV_PATH)
y_true = df[COL_TRUE].astype(str)
y_pred = df[COL_PRED].astype(str)
labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
n = max(1, len(labels))

# Row-normalized (for chord thickness by per-row share)
cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
cm_norm = np.nan_to_num(cm_norm)

# Raw counts (for alluvial width)
cm_cnt  = confusion_matrix(y_true, y_pred, labels=labels)
total_cnt = cm_cnt.sum() if cm_cnt.sum() > 0 else 1

cm_df_norm = pd.DataFrame(cm_norm, index=labels, columns=labels)
cm_df_cnt  = pd.DataFrame(cm_cnt,  index=labels, columns=labels)

# ========================= HEATMAP =======================
mint_teal_plus = LinearSegmentedColormap.from_list(
    "mint_teal_plus",
    ["#ffffff","#f6fbfa","#e9f5f1","#d9ede6","#c7e3d9",
     "#aed6ca","#92c6b7","#74b1a3","#5a9b90","#3f837b"],
    N=256
)
mint_teal_plus.set_under("#ffffff")
norm = PowerNorm(gamma=0.55, vmin=1e-3, vmax=1.0)
sns.set_theme(style="white")

cell_in = 0.28
pad_in  = 2.2
fig_side = float(np.clip(cell_in * n + pad_in, 6.0, 11.0))
fig, ax = plt.subplots(figsize=(fig_side, fig_side), constrained_layout=True)

hm = sns.heatmap(
    cm_df_norm, ax=ax,
    cmap=mint_teal_plus, norm=norm, cbar=False,
    linewidths=0.8, linecolor="white",
    square=True
)
ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
ax.set_xlabel("Predicted species", labelpad=4)
ax.set_ylabel("Reference species")
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="bottom")
plt.setp(ax.get_yticklabels(), rotation=0)

# Make heatmap tick labels bold (optional per your ask)
for t in ax.get_xticklabels(): t.set_fontweight('bold')
for t in ax.get_yticklabels(): t.set_fontweight('bold')

fig.canvas.draw()
pos = ax.get_position()
first_col_center_fig_x = pos.x0 + pos.width * (0.5 / n)
width_frac  = float(np.clip(2.5 / n, 0.12, 0.26))
height_frac = 0.018
gap_top     = 0.05
w = pos.width * width_frac
h = height_frac
left   = first_col_center_fig_x - 0.5 * w
bottom = pos.y1 + gap_top
cax = fig.add_axes([left, bottom, w, h])
sm = plt.cm.ScalarMappable(norm=norm, cmap=mint_teal_plus)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=7, pad=0.5)
cb.set_ticks([0.0, 0.5, 1.0]); cb.set_ticklabels(["0", "0.5", "1"])
for spine in cax.spines.values(): spine.set_visible(False)

plt.savefig(OUT_HM_PNG, bbox_inches="tight", dpi=1500, transparent=True)
plt.close(fig)

# ========================= COLORS ========================
base_colors = sns.color_palette("Set2", n)
def darken(rgb, factor=0.9): return tuple(max(0.0, min(1.0, c*factor)) for c in rgb)
node_colors = [darken(c, 0.9) for c in base_colors]

# ========================= HELPERS =======================
def filter_topk_threshold(mat_norm, min_flow=MIN_FLOW_NORM, top_k=TOP_K_PER_ROW):
    """Return list of (i,j,val_norm) passing threshold and optional top-k per row."""
    edges = []
    for i in range(mat_norm.shape[0]):
        row = mat_norm[i].copy()
        idx = np.where(row >= min_flow)[0]
        if top_k is not None and len(idx) > 0:
            top_idx = np.argsort(row)[::-1][:top_k]
            idx = np.intersect1d(idx, top_idx)
        for j in idx:
            v = float(row[j])
            if v > 0:
                edges.append((i, j, v))
    return edges

def bezier_ribbon(x0, y0a, y0b, x1, y1a, y1b, curvature=0.5, color=(0,0,0), alpha=0.6, lw=0.5):
    """Smooth ribbon between [y0a,y0b] at x0 and [y1a,y1b] at x1 using cubic Beziers."""
    cx0 = x0 + curvature * (x1 - x0)
    cx1 = x1 - curvature * (x1 - x0)
    verts = [
        (x0, y0b), (cx0, y0b), (cx1, y1b), (x1, y1b),
        (x1, y1a), (cx1, y1a), (cx0, y0a), (x0, y0a), (x0, y0b)
    ]
    codes = [
        Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY
    ]
    path = Path(verts, codes)
    return PathPatch(path, facecolor=color, edgecolor="white", lw=lw, alpha=alpha)

# ========================= CHORD (ticks+labels, NO big bars) =================
def plot_chord_ticks_labels_no_ring(mat_norm,
                                    labels,
                                    colors,
                                    out_png,
                                    min_flow=0.0,        # draw ALL connections
                                    width_scale=12.0,
                                    tick_len=0.012,      # small ticks
                                    r_pad=0.018,         # <<< slightly larger gap beyond tick tip
                                    label_fontsize=10,    # 5 pt
                                    label_weight="bold", # <<< make labels bold
                                    edge_alpha=0.5,
                                    curvature_pull=0.6):
    """
    Circular chord-style plot:
      - NO outer ring / big arcs
      - KEEP small ticks + labels
      - DRAW ALL connections (min_flow=0.0)
    """
    n = len(labels)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = np.c_[np.cos(theta), np.sin(theta)]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    # ticks + labels (aligned radially with ticks)
    for k, (x, y) in enumerate(pos):
        # tick (radial stroke)
        ax.plot([(1.0 - tick_len)*x, (1.0 + tick_len)*x],
                [(1.0 - tick_len)*y, (1.0 + tick_len)*y],
                color=colors[k], lw=2)

        ang_deg = np.degrees(np.arctan2(y, x))
        # label just beyond tick tip on same radial, with a bit more outer padding
        lx = (1.0 + tick_len + r_pad) * x
        ly = (1.0 + tick_len + r_pad) * y

        # keep text upright on left hemisphere
        if -90 <= ang_deg <= 90:
            rot = ang_deg;  ha = "left"
        else:
            rot = ang_deg + 180;  ha = "right"

        ax.text(lx, ly, labels[k],
                rotation=rot, rotation_mode="anchor",
                ha=ha, va="center",
                color=colors[k], fontsize=label_fontsize,
                fontweight=label_weight,  # bold labels
                clip_on=False)

    # connections (colored by source node)
    edges = filter_topk_threshold(mat_norm, min_flow=min_flow, top_k=TOP_K_PER_ROW)
    for i, j, v in edges:
        p0 = pos[i]; p1 = pos[j]
        c0 = p0 * curvature_pull
        c1 = p1 * curvature_pull
        lw = max(0.2, width_scale * v)
        path = Path([p0, c0, c1, p1],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        patch = PathPatch(path, facecolor="none",
                          edgecolor=colors[i], lw=lw,
                          alpha=edge_alpha, capstyle="round", joinstyle="round")
        ax.add_patch(patch)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=1500, transparent=True)
    plt.close(fig)

# ========================= ALLUVIAL (LEFT→RIGHT) =========
def plot_alluvial(cm_counts, labels, colors, out_png, min_flow_norm=0.02):
    """
    Alluvial using raw counts: left = actual, right = predicted.
    Ribbon width ∝ count. Layout spans y ∈ [0,1].
    """
    n = len(labels)
    total = cm_counts.values.sum()
    if total == 0: total = 1

    left_sizes  = cm_counts.sum(axis=1).values / total
    right_sizes = cm_counts.sum(axis=0).values / total

    left_starts  = np.r_[0, np.cumsum(left_sizes[:-1])]
    left_ends    = left_starts + left_sizes
    right_starts = np.r_[0, np.cumsum(right_sizes[:-1])]
    right_ends   = right_starts + right_sizes

    left_ptr  = left_starts.copy()
    right_ptr = right_starts.copy()

    fig_w, fig_h = 12, max(6, min(10, 0.3 * n + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    xL, xR = 0.08, 0.92
    bar_w = 0.015

    for i in range(n):
        ax.add_patch(Rectangle((xL - bar_w/2, left_starts[i]),
                               bar_w, left_sizes[i],
                               facecolor=colors[i], edgecolor="white", lw=0.6))
        ax.text(xL - 0.03, (left_starts[i] + left_ends[i]) / 2,
                labels[i], ha="right", va="center", color=colors[i], fontsize=10, fontweight="bold")

    for j in range(n):
        ax.add_patch(Rectangle((xR - bar_w/2, right_starts[j]),
                               bar_w, right_sizes[j],
                               facecolor=colors[j], edgecolor="white", lw=0.6))
        ax.text(xR + 0.03, (right_starts[j] + right_ends[j]) / 2,
                labels[j], ha="left", va="center", color=colors[j], fontsize=10, fontweight="bold")

    cm_norm_total = (cm_counts / total).values
    for i in range(n):
        row = cm_norm_total[i]
        idx = np.where(row >= min_flow_norm)[0]
        if TOP_K_PER_ROW is not None and len(idx) > 0:
            top_idx = np.argsort(row)[::-1][:TOP_K_PER_ROW]
            idx = np.intersect1d(idx, top_idx)

        for j in idx:
            frac = row[j]
            if frac <= 0: continue
            h = frac
            y0a, y0b = left_ptr[i], left_ptr[i] + h
            y1a, y1b = right_ptr[j], right_ptr[j] + h
            left_ptr[i]  += h
            right_ptr[j] += h

            ribbon = bezier_ribbon(
                x0=xL + bar_w/2, y0a=y0a, y0b=y0b,
                x1=xR - bar_w/2, y1a=y1a, y1b=y1b,
                curvature=0.5, color=colors[i], alpha=0.55, lw=0.4
            )
            ax.add_patch(ribbon)

    ax.text(0.5, 0.98, "Alluvial of Actual → Predicted (width ∝ count)",
            ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.015, f"Flows shown where fraction ≥ {min_flow_norm:.2f}"
                        + (f", top-{TOP_K_PER_ROW} per row" if TOP_K_PER_ROW else ""),
            ha="center", va="bottom", color=(0.3,0.3,0.3), fontsize=10)

    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight", dpi=1500, transparent=True); plt.close(fig)

# ========================= RUN ===========================
if __name__ == "__main__":
    # Chord: NO big bars/ring, KEEP small ticks + bold labels slightly outside ticks, KEEP ALL connections
    plot_chord_ticks_labels_no_ring(cm_df_norm.values, labels, node_colors, OUT_CHORD_PNG,
                                    min_flow=MIN_FLOW_NORM,  # 0.0 => draw all >0
                                    width_scale=12.0,
                                    tick_len=0.012, r_pad=0.15,   # moved outward vs 0.008
                                    label_fontsize=10, label_weight="bold",
                                    edge_alpha=0.55, curvature_pull=0.6)

    # Alluvial: volume-true ribbons
    plot_alluvial(cm_df_cnt, labels, node_colors, OUT_ALLUVIAL, min_flow_norm=0.02)

    print(f"Saved:\n  {OUT_HM_PNG}\n  {OUT_CHORD_PNG}\n  {OUT_ALLUVIAL}")
