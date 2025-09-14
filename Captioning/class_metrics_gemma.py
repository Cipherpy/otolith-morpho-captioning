import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

CSV = "/home/reshma/Otolith/captioning/otolith/llama/output_filtered.csv"
LABEL_COL_TRUE = "actual_label"
LABEL_COL_PRED = "predicted_label"

# If predicted has labels not in ground truth, collapse them into one bucket?
COLLAPSE_EXTRA_TO_OTHER = False   # set True to add a single "Other/OOD" column

df = pd.read_csv(CSV)
y_true = df[LABEL_COL_TRUE].astype(str)
y_pred = df[LABEL_COL_PRED].astype(str)

true_set = set(y_true.unique())
pred_set = set(y_pred.unique())

if COLLAPSE_EXTRA_TO_OTHER:
    OTHER = "Other/OOD"
    y_pred = y_pred.where(y_pred.isin(true_set), OTHER)
    labels = sorted(true_set) + ([OTHER] if OTHER in y_pred.values else [])
else:
    labels = sorted(true_set.union(pred_set))

# Metrics (optional printout)
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}\n")
print(classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=0))

# Confusion matrix (counts)
cm = confusion_matrix(y_true, y_pred, labels=labels)

# --- PLOT: counts only, ORANGE ---
n = len(labels)
fig_w = max(10, 0.45 * n + 6)
plt.figure(figsize=(fig_w, 8), dpi=150)
sns.set(style="white")

ax = sns.heatmap(
    cm,
    annot=True, fmt="d",
    cmap="Oranges",      # << orange colormap
    xticklabels=labels, yticklabels=labels,
    cbar=True, linewidths=0.5, linecolor="white",
    annot_kws={"size": 8}
)
#ax.set_title("Confusion Matrix (Counts)", fontweight="bold")
ax.set_xlabel("Identified Species")
ax.set_ylabel("Reference Species")
plt.xticks(rotation=75)
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("llama/plots/confusion_matrix_counts_orange_transparent.png", bbox_inches="tight", dpi=600, transparent=True)
plt.show()
