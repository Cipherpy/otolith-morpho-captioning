
#!/usr/bin/env python3
"""
Compute BLEU-1 … BLEU-4, METEOR, CIDEr, and SPICE for
generated vs. ground-truth captions.
"""

import os, re
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from rouge_score import rouge_scorer
from tqdm import tqdm

df=pd.read_csv("/home/reshma/Otolith/captioning/otolith/paired_captions_minimal.csv")
#os.environ["JAVA_TOOL_OPTIONS"] = "--add-opens java.base/java.lang=ALL-UNNAMED"


STRIP_CLASS_TAG = True                  # remove trailing “Class detected: …”
if STRIP_CLASS_TAG:
    rx = re.compile(r"\s*Class(?:\s+detected)?\s*:.*$", flags=re.I | re.S)
    df["hyp"] = df["generated_caption"].astype(str).apply(lambda s: re.sub(rx, "", s).strip())

# ---------------------------------------------------------------------
# 3.  Metric objects
# ---------------------------------------------------------------------
nltk.download("punkt", quiet=True)
rouge   = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
smooth   = SmoothingFunction().method1
meteor   = Meteor()
cider    = Cider()
#spice    = Spice()

bleu1s, bleu2s, bleu3s, bleu4s = [], [], [], []
gts, res = {}, {}
rouges = []          # <-- FIX: this must be a single list

# ------------------------------------------------------------
# 4.  Loop over caption pairs
# ------------------------------------------------------------
print("[INFO] Scoring sentence-level ROUGE-L …")
for idx, r in tqdm(df.iterrows(), total=len(df), unit="caption"):
    ref, hyp = r["Description"], r["generated_caption"]

    # for corpus-level metrics (if enabled later)
    gts[str(idx)] = [ref]
    res[str(idx)] = [hyp]
    # ROUGE-L (sentence-level F-measure)
    rouges.append(rouge.score(ref, hyp)["rougeL"].fmeasure)

    ref_tok = [nltk.word_tokenize(ref.lower())]
    hyp_tok = nltk.word_tokenize(hyp.lower())

    bleu1s.append(sentence_bleu(ref_tok, hyp_tok, weights=(1, 0, 0, 0), smoothing_function=smooth))
    bleu2s.append(sentence_bleu(ref_tok, hyp_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
    bleu3s.append(sentence_bleu(ref_tok, hyp_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
    bleu4s.append(sentence_bleu(ref_tok, hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

# ---------------------------------------------------------------------
# 5.  Corpus-level metrics
# ---------------------------------------------------------------------

meteor_score, _ = meteor.compute_score(gts, res)
cider_score,  _ = cider.compute_score(gts, res)
#spice_score,  _ = spice.compute_score(gts, res)

# ---------------------------------------------------------------------
# 6.  Report
# ---------------------------------------------------------------------
print("\n=== Caption quality metrics ===")
print(f"BLEU-1 : {sum(bleu1s)/len(bleu1s):.4f}")
print(f"BLEU-2 : {sum(bleu2s)/len(bleu2s):.4f}")
print(f"BLEU-3 : {sum(bleu3s)/len(bleu3s):.4f}")
print(f"BLEU-4 : {sum(bleu4s)/len(bleu4s):.4f}")
print(f"METEOR : {meteor_score:.4f}")
print(f"CIDEr  : {cider_score:.4f}")
print(f"ROUGE-L: {sum(rouges) / len(rouges):.4f}")
#print(f"SPICE  : {spice_score:.4f}")