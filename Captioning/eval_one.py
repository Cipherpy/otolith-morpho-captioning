#!/usr/bin/env python3
"""
Evaluate every caption pair in a CSV with

  • BLEU-1 … BLEU-4
  • ROUGE-L
  • METEOR  (COCO jar)
  • CIDEr   (COCO)

Outputs
-------
caption_scores_all.csv   – per-image metrics
stdout                   – averaged BLEU/ROUGE and corpus-level METEOR & CIDEr
"""

import re
import pandas as pd
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score   import sentence_bleu, SmoothingFunction
from rouge_score                 import rouge_scorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider   import Cider

# ---------------- Config -----------------
CSV_PATH  = "/home/reshma/Otolith/captioning/otolith/llama/llama_OOD_output_filtered.csv"
OUT_CSV   = "llama/llama_OOD_otolith_paired_captions_scores.csv"
STRIP_TAG = True       # remove trailing “Class: …” from generated captions

# ---------------- Metric objects ---------
smooth  = SmoothingFunction().method1
rouge   = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
meteor  = Meteor()
cider   = Cider()

# ------------------------------------------------------------------
# 1.  Load data
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

if STRIP_TAG:
    rx = re.compile(r"\s*Class(?:\s+detected)?\s*:.*$", re.I | re.S)
    df["hyp"] = df["generated_caption"].astype(str).apply(
        lambda s: re.sub(rx, "", s).strip())
else:
    df["hyp"] = df["generated_caption"].astype(str)

# ------------------------------------------------------------------
# 2.  Evaluation loop
# ------------------------------------------------------------------
gts, res, per_row = {}, {}, []

# containers for mean calculations
bleu1s, bleu2s, bleu3s, bleu4s, rouge_vals = [], [], [], [], []

print(f"[INFO] Scoring {len(df)} caption pairs …")
for idx, row in tqdm(df.iterrows(), total=len(df), unit="pair"):

    ref_raw = re.sub(r"[\r\n\t]", " ", str(row["Description"]).strip())
    hyp_raw = re.sub(r"[\r\n\t]", " ", str(row["hyp"]).strip())

    # tokenise once
    ref_tok = nltk.word_tokenize(ref_raw.lower())
    hyp_tok = nltk.word_tokenize(hyp_raw.lower())

    # ------ BLEU ------
    b1 = sentence_bleu([ref_tok], hyp_tok, weights=(1, 0, 0, 0),            smoothing_function=smooth)
    b2 = sentence_bleu([ref_tok], hyp_tok, weights=(0.5, 0.5, 0, 0),        smoothing_function=smooth)
    b3 = sentence_bleu([ref_tok], hyp_tok, weights=(1/3, 1/3, 1/3, 0),      smoothing_function=smooth)
    b4 = sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=smooth)

    bleu1s.append(b1); bleu2s.append(b2); bleu3s.append(b3); bleu4s.append(b4)

    # ------ ROUGE-L ------
    rL = rouge.score(ref_raw, hyp_raw)["rougeL"].fmeasure
    rouge_vals.append(rL)

    # ------ METEOR & CIDEr (per-image) ------
    img_id = str(idx)
    m_single, _ = meteor.compute_score({img_id: [ref_raw]}, {img_id: [hyp_raw]})
    c_single, _ = cider .compute_score({img_id: [ref_raw]}, {img_id: [hyp_raw]})

    gts[img_id] = [ref_raw]
    res[img_id] = [hyp_raw]

    per_row.append({
        "Image"           : row.get("Image",        "--"),
        "actual_label"    : row.get("actual_label", "NA"),
        "predicted_label" : row.get("predicted_label", "NA"),
        "BLEU-1"          : b1,
        "BLEU-2"          : b2,
        "BLEU-3"          : b3,
        "BLEU-4"          : b4,
        "ROUGE-L"         : rL,
        "METEOR"          : m_single,
        "CIDEr"           : c_single,
    })

# ------------------------------------------------------------------
# 3.  Corpus-level METEOR/CIDEr
# ------------------------------------------------------------------
meteor_corpus, _ = meteor.compute_score(gts, res)
cider_corpus , _ = cider .compute_score(gts, res)

# ------------------------------------------------------------------
# 4.  Save per-image CSV
# ------------------------------------------------------------------
pd.DataFrame(per_row).to_csv(OUT_CSV, index=False)
print(f"\n[INFO] Per-pair scores saved to {OUT_CSV}")

# ------------------------------------------------------------------
# 5.  Pretty summary
# ------------------------------------------------------------------
print("\n=== Caption quality metrics ===")
print(f"BLEU-1 : {sum(bleu1s)/len(bleu1s):.4f}")
print(f"BLEU-2 : {sum(bleu2s)/len(bleu2s):.4f}")
print(f"BLEU-3 : {sum(bleu3s)/len(bleu3s):.4f}")
print(f"BLEU-4 : {sum(bleu4s)/len(bleu4s):.4f}")
print(f"METEOR : {meteor_corpus:.4f}")
print(f"CIDEr  : {cider_corpus :.4f}")
print(f"ROUGE-L: {sum(rouge_vals)/len(rouge_vals):.4f}")
