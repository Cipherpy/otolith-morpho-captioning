import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# === Load ===
df = pd.read_csv("/home/reshma/Otolith/otolith_Final/src/embedding_metrics_id_ood.csv")

# Basic sanity
required = {"path","source","true_label","assign_class","maha_min","pca_residual","knn_mean"}
assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
df["is_ood"] = (df["source"]=="OOD").astype(int)

# === 1) Summary by source ===
summ = df.groupby("source")[["maha_min","pca_residual","knn_mean"]].agg(
    ["count","mean","median","std",lambda x: np.quantile(x,0.25),lambda x: np.quantile(x,0.75)]
)
summ.columns = ["_".join(filter(None,c)).replace("<lambda_0>","q25").replace("<lambda_1>","q75") for c in summ.columns]
print("\nSummary by source:\n", summ)

# === 2) ID thresholds (95th/99th) ===
id_df = df[df["source"]=="Test"]
thr = {
    "maha_min":  {"p95": np.quantile(id_df["maha_min"],0.95), "p99": np.quantile(id_df["maha_min"],0.99)},
    "pca_residual":{"p95": np.quantile(id_df["pca_residual"],0.95), "p99": np.quantile(id_df["pca_residual"],0.99)},
    "knn_mean": {"p95": np.quantile(id_df["knn_mean"],0.95), "p99": np.quantile(id_df["knn_mean"],0.99)},
}
print("\nID-derived thresholds:\n", thr)

# === 3) OOD detection rates at thresholds ===
def det_rate(metric, q="p95"):
    t = thr[metric][q]
    det = (df[metric] > t).mean()    # overall
    det_ood = (df.loc[df.is_ood==1, metric] > t).mean()
    fp_id  = (df.loc[df.is_ood==0, metric] > t).mean()
    return det, det_ood, fp_id

rows=[]
for m in ["maha_min","pca_residual","knn_mean"]:
    for q in ["p95","p99"]:
        overall, ood, fp = det_rate(m,q)
        rows.append({"metric":m,"quantile":q,"ood_detect_rate":ood,"id_false_positive_rate":fp})
det_table = pd.DataFrame(rows).sort_values(["metric","quantile"])
print("\nDetection rates (higher ood_detect_rate and lower id_false_positive_rate are better):\n", det_table)

# === 4) Simple combined OOD score (z-avg) ===
# Standardize using ID stats
def zscore(col):
    mu, sd = id_df[col].mean(), id_df[col].std() + 1e-12
    return (df[col]-mu)/sd

df["z_maha"] = zscore("maha_min")
df["z_res"]  = zscore("pca_residual")
df["z_knn"]  = zscore("knn_mean")
df["ood_score"] = df[["z_maha","z_res","z_knn"]].mean(axis=1)

# AUC of each metric & combined
for col in ["maha_min","pca_residual","knn_mean","ood_score"]:
    auc = roc_auc_score(df["is_ood"], df[col])
    print(f"AUC({col}): {auc:.3f}")

# Pick a working point: 95th percentile of ID ood_score
ood_thr = np.quantile(df.loc[df.is_ood==0,"ood_score"], 0.95)
ood_det_rate = (df.loc[df.is_ood==1,"ood_score"] > ood_thr).mean()
id_fp_rate   = (df.loc[df.is_ood==0,"ood_score"] > ood_thr).mean()
print(f"\nCombined score @ID p95: OOD detect={ood_det_rate:.3f}, ID FP={id_fp_rate:.3f}, thr={ood_thr:.2f}")

# === 5) Alignment: which ID classes OOD gravitates to ===
ct = pd.crosstab(df.loc[df.is_ood==1,"true_label"], df.loc[df.is_ood==1,"assign_class"], normalize="index")
print("\nOOD→Assigned-class distribution (row-normalized):\n", ct.fillna(0).round(3))

# === 6) Top “inlier-like” OOD vs “outlier-like” ID ===
# Closest OOD to ID manifold (low maha, low residual, low knn): sort by ood_score asc
top_inlier_like_ood = df[df.is_ood==1].sort_values("ood_score").head(15)[
    ["path","true_label","assign_class","maha_min","pca_residual","knn_mean","ood_score"]
]
print("\nClosest OOD (might be confusable):\n", top_inlier_like_ood)

# Most outlier-ish ID (could be mislabeled or problematic imaging)
top_outlier_id = df[df.is_ood==0].sort_values("ood_score", ascending=False).head(15)[
    ["path","true_label","assign_class","maha_min","pca_residual","knn_mean","ood_score"]
]
print("\nMost outlier ID samples:\n", top_outlier_id)

# === 7) Save CSV snapshots ===
det_table.to_csv("ood_detection_table.csv", index=False)
top_inlier_like_ood.to_csv("top_inlier_like_OOD.csv", index=False)
top_outlier_id.to_csv("top_outlier_ID.csv", index=False)
ct.to_csv("ood_alignment_crosstab.csv")

# === 8) Minimal plots (no seaborn, one chart per fig, no explicit colors) ===
def save_hist(col, bins=40):
    plt.figure(figsize=(6,4))
    for src in ["Test","OOD"]:
        x = df.loc[df.source==src, col].values
        plt.hist(x, bins=bins, alpha=0.5, label=src, density=True)
    plt.xlabel(col); plt.ylabel("density"); plt.title(f"Distribution of {col}")
    plt.legend(); plt.tight_layout(); plt.savefig(f"hist_{col}.png", dpi=200); plt.close()

for m in ["maha_min","pca_residual","knn_mean","ood_score"]:
    save_hist(m)

# Pairwise scatter: OOD vs Test
def scatter_xy(xc, yc):
    plt.figure(figsize=(6,5))
    mask = (df.source=="Test").values
    plt.scatter(df.loc[mask, xc], df.loc[mask, yc], s=8, alpha=0.5, label="Test")
    plt.scatter(df.loc[~mask, xc], df.loc[~mask, yc], s=10, alpha=0.7, label="OOD", marker="x")
    plt.xlabel(xc); plt.ylabel(yc); plt.title(f"{xc} vs {yc}")
    plt.legend(); plt.tight_layout(); plt.savefig(f"scatter_{xc}_vs_{yc}.png", dpi=200); plt.close()

scatter_xy("maha_min","pca_residual")
scatter_xy("maha_min","knn_mean")
scatter_xy("pca_residual","knn_mean")

print("\nWritten files:")
print("- ood_detection_table.csv")
print("- top_inlier_like_OOD.csv")
print("- top_outlier_ID.csv")
print("- ood_alignment_crosstab.csv")
print("- hist_[metric].png & scatter_[x]_vs_[y].png")
