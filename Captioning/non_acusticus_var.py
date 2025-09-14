#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sulcus acusticus evaluation:
- Confusion matrix (raw strings, no normalization)
- Metrics for the "Not visible" class (recall, precision, F1)
- Saves figure (.png) and a metrics report (.txt)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Config
# =========================
CSV_PATH = "/home/reshma/Otolith/captioning/otolith/llama/output_filtered.csv"  # <-- update if needed
OUT_DIR  = "llama/plots"
CM_FIG   = os.path.join(OUT_DIR, "sulcus_acusticus_confusion_matrix_seaborn.png")
METRICS_TXT = os.path.join(OUT_DIR, "sulcus_not_visible_metrics.txt")


# =========================
# Helpers
# =========================
def extract_raw_sulcus_acusticus(text: str) -> str:
    """Extract the exact raw text after 'Sulcus acusticus:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r"Sulcus acusticus:\s*([^\.]*)", str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

def is_not_visible(s: str) -> bool:
    """True if string mentions 'not visible' (any qualifiers allowed)."""
    return isinstance(s, str) and re.search(r"\bnot\s*visible\b", s, flags=re.IGNORECASE) is not None


# =========================
# Load & prep
# =========================
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

df["sulcus_acusticus_gt_raw"]  = df["Description"].apply(extract_raw_sulcus_acusticus)
df["sulcus_acusticus_gen_raw"] = df["generated_caption"].apply(extract_raw_sulcus_acusticus)

filtered = df[(df["sulcus_acusticus_gt_raw"] != "") & (df["sulcus_acusticus_gen_raw"] != "")].copy()


# =========================
# Confusion matrix (raw)
# =========================
conf = pd.crosstab(
    filtered["sulcus_acusticus_gt_raw"],
    filtered["sulcus_acusticus_gen_raw"],
    rownames=["Ground Truth Sulcus acusticus"],
    colnames=["Predicted Sulcus acusticus"],
    dropna=False
)

# Sort rows/cols by total frequency for readability
row_order = conf.sum(axis=1).sort_values(ascending=False).index
col_order = conf.sum(axis=0).sort_values(ascending=False).index
conf = conf.loc[row_order, col_order]

# ---- seaborn plot
sns.set_theme(style="white")
fig_w = max(14, 0.75 * len(conf.columns))
fig_h = max(12, 0.60 * len(conf.index))
plt.figure(figsize=(fig_w, fig_h))

ax = sns.heatmap(
    conf,
    annot=True,
    fmt="d",
    cmap="cividis",               # try: 'viridis', 'YlGnBu', 'Blues'
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "Count"},
    square=False
)

ax.set_title("Confusion Matrix: Sulcus acusticus (Raw strings)", fontsize=18, pad=18)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Ground Truth", fontsize=12)

# Tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

# Improve annotation contrast (white on dark cells)
vals = conf.values
thr = vals.max() * 0.5 if vals.size else 0
for t in ax.texts:
    try:
        val = int(t.get_text())
    except ValueError:
        val = 0
    t.set_color("white" if val > thr else "black")
    t.set_fontsize(8)

plt.tight_layout()
plt.savefig(CM_FIG, dpi=600, bbox_inches="tight")
plt.show()


# =========================
# "Not visible" metrics
# =========================
both = filtered.copy()
both["gt_nv"]   = both["sulcus_acusticus_gt_raw"].apply(is_not_visible)
both["pred_nv"] = both["sulcus_acusticus_gen_raw"].apply(is_not_visible)

total_gt_nv   = int(both["gt_nv"].sum())                     # support of Not visible in GT
total_pred_nv = int(both["pred_nv"].sum())                   # how many predictions say Not visible
correct_nv    = int((both["gt_nv"] & both["pred_nv"]).sum()) # true positives

recall_nv    = correct_nv / total_gt_nv   if total_gt_nv   else 0.0
precision_nv = correct_nv / total_pred_nv if total_pred_nv else 0.0
f1_nv = (2*precision_nv*recall_nv)/(precision_nv+recall_nv) if (precision_nv+recall_nv) else 0.0

# Top wrong predictions when GT is Not visible
mistakes = both[both["gt_nv"] & ~both["pred_nv"]]
top_wrong = mistakes["sulcus_acusticus_gen_raw"].value_counts().head(10)

# Print short summary
print("\nSulcus acusticus — 'Not visible' metrics")
print(f"  Support (GT Not visible): {total_gt_nv}")
print(f"  Correctly predicted:      {correct_nv}")
print(f"  Recall:                    {recall_nv:.2%}")
print(f"  Precision:                 {precision_nv:.2%}")
print(f"  F1 score:                  {f1_nv:.3f}")

if not top_wrong.empty:
    print("\nTop predicted strings when GT is 'Not visible' (errors):")
    for k, v in top_wrong.items():
        print(f"  {v:4d}  {k}")

# Write full report to file
with open(METRICS_TXT, "w", encoding="utf-8") as f:
    f.write("Sulcus acusticus — 'Not visible' metrics\n")
    f.write(f"Support (GT Not visible): {total_gt_nv}\n")
    f.write(f"Correctly predicted:      {correct_nv}\n")
    f.write(f"Recall:                    {recall_nv:.2%}\n")
    f.write(f"Precision:                 {precision_nv:.2%}\n")
    f.write(f"F1 score:                  {f1_nv:.3f}\n\n")
    if not top_wrong.empty:
        f.write("Top predicted strings when GT is 'Not visible' (errors):\n")
        for k, v in top_wrong.items():
            f.write(f"{v:4d}  {k}\n")

print(f"\nSaved confusion matrix: {CM_FIG}")
print(f"Saved metrics report:   {METRICS_TXT}")
