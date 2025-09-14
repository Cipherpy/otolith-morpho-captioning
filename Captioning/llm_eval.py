#!/usr/bin/env python3
"""
Evaluate generated image captions against ground-truth captions
using GPT.  Scores are 1–10 with a short justification.

Dependencies:
  pip install --upgrade openai pandas tqdm python-dotenv

Set your API key (one of):
  export OPENAI_API_KEY="sk-..."
  or create a .env file with OPENAI_API_KEY=sk-...
"""

import os, re, time, json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 0.  Config – adjust paths / model as needed
# ------------------------------------------------------------------
#PATH_GEN = "otolith_test_results.xlsx"      # generated captions
PATH_GT  = "/home/reshma/Otolith/captioning/otolith/llama/llama_OOD_output_filtered.csv"
##OUT_PAIRED  = "paired_captions.csv"
OUT_SCORED  = "llama/llama_OOD_GPT_scored_captions.csv"
MODEL_NAME  = "gpt-4o"                 # or gpt-4o / gpt-4-turbo / gpt-3.5-turbo etc.
MAX_RETRIES = 3

# ------------------------------------------------------------------
# 1.  Initialise OpenAI client
# ------------------------------------------------------------------
load_dotenv()                               # loads OPENAI_API_KEY from .env if present
client = OpenAI()                           # expects env var

# ------------------------------------------------------------------
# 2.  Load and align the two caption files
# ------------------------------------------------------------------
df = pd.read_csv(PATH_GT)            # has 'image_path', 'generated_caption'

# ------------------------------------------------------------------
# 3.  Cleaning helper – strip trailing “Class detected: …”
# ------------------------------------------------------------------
CLASS_TAG_RX = re.compile(r"\s*Class(?:\s+detected)?\s*:.*$", flags=re.I | re.S)

def strip_class_tag(text: str) -> str:
    """Remove 'Class detected: …' or 'Class: …' from caption."""
    return re.sub(CLASS_TAG_RX, "", str(text)).strip()

# ------------------------------------------------------------------
# 4.  GPT evaluation helper
# ------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert evaluator of scientific image captions.
You receive:
  GROUND-TRUTH: the correct, human-written description
  GENERATED:    a candidate description
Return **JSON** with:
  "score":  integer 1–10  (10 = perfect match, 1 = completely wrong)
  "reason": one short sentence explaining the score.
Scoring guideline:
  10 = identical or wording differences only
   7–9 = minor factual differences
   4–6 = partly correct but important omissions / errors
   1–3 = mostly or entirely wrong"""

def grade_pair(ref: str, hyp: str, retries: int = MAX_RETRIES) -> dict:
    """Call the OpenAI Chat Completion API and return {'score':int,'reason':str}."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"GROUND-TRUTH:\n{ref}\n\nGENERATED:\n{hyp}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            wait = 2 ** attempt
            print(f"[WARN] API error ({e}) – retrying in {wait}s…")
            time.sleep(wait)
    return {"score": 0, "reason": "API failed after retries"}

# ------------------------------------------------------------------
# 5.  Score each pair
# ------------------------------------------------------------------
scores, reasons = [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="GPT grading"):
    ref = row["Description"]
    hyp = strip_class_tag(row["generated_caption"])
    result = grade_pair(ref, hyp)
    scores.append(result.get("score", 0))
    reasons.append(result.get("reason", "n/a"))

df["gpt_score"]  = scores
df["gpt_reason"] = reasons

# ------------------------------------------------------------------
# 6.  Save results and print summary
# ------------------------------------------------------------------
df.to_csv(OUT_SCORED, index=False)
print(f"[INFO] Scored captions written to {OUT_SCORED}")

avg_score = df["gpt_score"].mean()
print(f"\nAverage GPT score over {len(df)} captions: {avg_score:.2f} / 10")
