#!/usr/bin/env python3
"""
Compute BLEU-1 … BLEU-4, METEOR, CIDEr, and SPICE for
generated vs. ground-truth captions, while also retaining
the ground-truth class/label for each image.
"""
import os, re
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# ---------------------------------------------------------------------
# 0.  Config – adjust paths & cleaning options
# ---------------------------------------------------------------------
PATH_GEN = "/home/reshma/Otolith/captioning/otolith/grayscale_otolith_test_results_progress.csv"   # cols: image_path, generated_caption
PATH_GT  = "/home/reshma/Otolith/captioning/otolith/text_data/test_modified.csv"  # cols: Image, Description, Class
GT_CLASS_COL = "Class"                  # name of the class/label column in PATH_GT
STRIP_CLASS_TAG = True                  # remove trailing “Class detected: …”

# ---------------------------------------------------------------------
# 1.  Load & align
# ---------------------------------------------------------------------
df_gen = pd.read_csv(PATH_GEN)
df_gt  = pd.read_csv(PATH_GT)

# unify file names
df_gen["Image"] = df_gen["image_path"].apply(os.path.basename)

# Merge, keeping the class column
cols_from_gt = ["Image", "Description"]
if GT_CLASS_COL in df_gt.columns:
    cols_from_gt.append(GT_CLASS_COL)

df = pd.merge(
    df_gen[["Image", "generated_caption"]],
    df_gt[cols_from_gt],
    on="Image",
    how="inner"
).rename(columns={
    "generated_caption": "hyp",
    "Description":       "ref"
})

print(f"[INFO] Paired rows: {len(df)}")

# Optional caption cleanup
if STRIP_CLASS_TAG:
    rx = re.compile(r"\s*Class(?:\s+detected)?\s*:.*$", flags=re.I | re.S)
    df["hyp"] = df["hyp"].astype(str).apply(lambda s: re.sub(rx, "", s).strip())

# ---------------------------------------------------------------------
# 2.  Save paired dataframe
# ---------------------------------------------------------------------
out_cols = ["Image", "ref", "hyp"]
if GT_CLASS_COL in df.columns:
    out_cols.insert(1, GT_CLASS_COL)   # keep class right after Image

out_path = "grayscale_paired_captions_with_class.csv"
df[out_cols].to_csv(out_path, index=False)
print(f"[INFO] Paired file written to: {out_path}")

# ---------------------------------------------------------------------
# 3.  (Optional) Continue with metric computation here...
# ---------------------------------------------------------------------
# ... your metric code can read `df` directly
