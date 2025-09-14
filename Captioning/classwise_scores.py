import pandas as pd

# Load your caption scores file
df = pd.read_csv("/home/reshma/Otolith/captioning/otolith/llama/llama_OOD_otolith_paired_captions_scores.csv")

# Check the first few rows
print(df.head())

# Group by actual class and compute mean scores
class_avg = df.groupby("actual_label")[["BLEU-1","BLEU-2","BLEU-3","BLEU-4",
                                        "ROUGE-L","METEOR","CIDEr"]].mean()

# Sort classes by average CIDEr (or any metric of interest)
class_avg_sorted = class_avg.sort_values("CIDEr", ascending=False)

# Display results
print(class_avg_sorted)

# If you want to save the summary
class_avg.to_csv("/home/reshma/Otolith/captioning/otolith/llama/llama_ood_classwise_avg_scores.csv")
