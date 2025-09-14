import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Inputs ----
GEMMA_CSV = "/home/reshma/Otolith/captioning/otolith/GRayscale_otolith_paired_captions_scores.csv"
LLAMA_CSV = "/home/reshma/Otolith/captioning/otolith/llama/Grayscale_llama_otolith_paired_captions_scores.csv"
LABEL_COL = "actual_label"
METRICS = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L']

# If you have many classes, you can show only the top-K by mean score to avoid clutter.
TOP_K = None   # e.g., 30 or None for all

# ---- Helpers ----
def prepare_scores(csv_path, label_col=LABEL_COL, metrics=METRICS):
    df = pd.read_csv(csv_path)

    # keep only metrics that exist
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError(f"No expected metric columns found in {csv_path}")

    # coerce to numeric
    df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

    # per-row mean
    df['auto_avg'] = df[metrics].mean(axis=1)

    # normalize 0–1 (per-file); guard divide-by-zero
    maxv = df['auto_avg'].max()
    df['auto_avg_norm'] = df['auto_avg'] / (maxv if maxv and maxv > 0 else 1.0)

    # group by class
    out = df.groupby(label_col, as_index=False)['auto_avg_norm'].mean()
    return out

# ---- Load & merge (intersection of classes to keep bars aligned) ----
gemma = prepare_scores(GEMMA_CSV).rename(columns={'auto_avg_norm': 'Gemma-3'})
llama  = prepare_scores(LLAMA_CSV).rename(columns={'auto_avg_norm': 'LLaMA 3.2'})

merged = pd.merge(gemma, llama, on=LABEL_COL, how='inner')  # use 'outer' and fillna(0) if you want all classes
if merged.empty:
    raise ValueError("No overlapping classes between the two CSVs. Try how='outer' and fillna(0).")

# Optional: limit to top-K by mean score (to reduce clutter)
merged['mean'] = merged[['Gemma-3', 'LLaMA 3.2']].mean(axis=1)
merged = merged.sort_values('mean', ascending=False)
if TOP_K is not None:
    merged = merged.head(TOP_K)

# ---- Plot ----
n = len(merged)
x = np.arange(n)
width = 0.44

fig_w = max(12, 0.45 * n + 6)
plt.figure(figsize=(fig_w, 7), dpi=120)

plt.bar(x - width/2, merged['Gemma-3'].values, width, label='Gemma-3')
plt.bar(x + width/2, merged['LLaMA 3.2'].values, width, label='LLaMA 3.2')

plt.xticks(x, merged[LABEL_COL].values, rotation=75, ha='right', fontsize=11)
plt.ylabel('Normalized score (0–1)', fontsize=12)
plt.xlabel('Species', fontsize=12)
plt.ylim(0, 1)
plt.legend(fontsize=11)
plt.tight_layout()

# ---- Save ----
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/grayscale_gemma_vs_llama_grouped_bar.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
