import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_raw_ostium(text):
    """Extract the exact raw text after 'Ostium:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r'Ostium:\s*([^\.]*)', str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

# ---- load & prep
df = pd.read_csv("/home/reshma/Otolith/captioning/otolith/llama/output_filtered.csv")

df["ostium_gt_raw"]  = df["Description"].apply(extract_raw_ostium)
df["ostium_gen_raw"] = df["generated_caption"].apply(extract_raw_ostium)

filtered = df[(df["ostium_gt_raw"] != "") & (df["ostium_gen_raw"] != "")]

conf = pd.crosstab(
    filtered["ostium_gt_raw"],
    filtered["ostium_gen_raw"],
    rownames=["Ground Truth Ostium"],
    colnames=["Predicted Ostium"],
    dropna=False
)

# ---- seaborn plot
sns.set_theme(style="white")
os.makedirs("llama/plots", exist_ok=True)

# dynamic figure size (larger than before)
fig_w = max(14, 0.75 * len(conf.columns))
fig_h = max(12, 0.60 * len(conf.index))
plt.figure(figsize=(fig_w, fig_h))

ax = sns.heatmap(
    conf,
    annot=True,
    fmt="d",
    cmap="cividis",              # try 'viridis', 'cividis', 'Blues', etc.
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "Count"},
    square=False
)

#ax.set_title("Confusion Matrix: Raw Cauda Values", fontsize=18, pad=18)
ax.set_xlabel("Predictions", fontsize=12)
ax.set_ylabel("Ground truth", fontsize=12)

# tick label formatting
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

# improve annotation contrast (white on dark cells)
texts = ax.texts
vals = conf.values
thr = vals.max() * 0.5 if vals.size else 0
for t in texts:
    try:
        val = int(t.get_text())
    except ValueError:
        val = 0
    t.set_color("white" if val > thr else "white")
    t.set_fontsize(9)

plt.tight_layout()
plt.savefig("llama/plots/ostium_confusion_matrix_seaborn.png", dpi=600, bbox_inches="tight")
plt.show()
