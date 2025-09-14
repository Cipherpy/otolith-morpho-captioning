import os
import pandas as pd

PATH_GEN = "/home/reshma/Otolith/captioning/otolith/llama/llama_OOD.csv"
PATH_GT  = "/home/reshma/Otolith/captioning/otolith/text_data/OOD_modified.csv"
OUT_PATH = "/home/reshma/Otolith/captioning/otolith/llama/llama_OOD_paired_captions_minimal.csv"

# Load data
df_gen = pd.read_csv(PATH_GEN)
df_gt = pd.read_csv(PATH_GT)

# Extract image name for joining
df_gen["Image"] = df_gen["image_path"].apply(os.path.basename)

# Select required columns (no image_path)
df_gen_sel = df_gen[["Image", "actual_label", "predicted_label", "generated_caption"]]
df_gt_sel = df_gt[["Image", "Description"]]

# Merge on image name
df_merged = pd.merge(df_gen_sel, df_gt_sel, on="Image", how="inner")

# Arrange columns as required: Image (filename), actual_label, predicted_label, generated_caption, Description
df_merged = df_merged[["Image", "actual_label", "predicted_label", "generated_caption", "Description"]]

# Save to CSV
df_merged.to_csv(OUT_PATH, index=False)
print(f"[INFO] File written to: {OUT_PATH}")

# Optional: Preview
print(df_merged.head())
