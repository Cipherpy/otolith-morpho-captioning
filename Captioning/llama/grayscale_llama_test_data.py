import torch
from unsloth import FastVisionModel
import pandas as pd
from PIL import Image
import os
import csv
from tqdm import tqdm

# --- Read user prompt ---
with open("/home/reshma/Otolith/captioning/otolith/prompts/user_prompt.txt", "r") as f:
    user_prompt = f.read()

# --- Model and Tokenizer Loading ---
model_path = "/gpu_das_hdd/reshma/llama/checkpoint-6350"
print("Loading fine-tuned model...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_path,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)
model.eval()
print("Model loaded for inference!")

# --- Paths ---
test_base_dir = "/home/reshma/Otolith/captioning/otolith/model_data/test"
out_csv = "/home/reshma/Otolith/captioning/otolith/llama/grayscale_llama_test_results_progress.csv"
fieldnames = ["image_path", "actual_label", "predicted_label", "generated_caption"]

# --- Output CSV Setup ---
if not os.path.isfile(out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# --- Helper to extract predicted class from caption ---
import re
def extract_predicted_label(caption):
    m = re.search(r"Class(?: detected)?:\s*(.+)", str(caption))
    if m:
        return m.group(1).strip()
    return ""

# --- Main loop over subfolders and images ---
for species in sorted(os.listdir(test_base_dir)):
    species_dir = os.path.join(test_base_dir, species)
    if not os.path.isdir(species_dir):
        continue

    for img_file in sorted(os.listdir(species_dir)):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        img_path = os.path.join(species_dir, img_file)

        try:
            image = Image.open(img_path).convert("L").convert("RGB")
        except Exception as e:
            print(f"✖  {img_path}: {e}")
            continue

        # Compose prompt for the model
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        # Generate caption
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    streamer=None,
                    max_new_tokens=256,
                    use_cache=False,
                    temperature=0.8,
                    min_p=0.1
                )
                generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"✖  Error generating caption for {img_path}: {e}")
            generated_caption = ""

        # Extract predicted label from generated caption
        pred_label = extract_predicted_label(generated_caption)

        # Prepare row
        row = dict(
            image_path=img_path,
            actual_label=species,
            predicted_label=pred_label,
            generated_caption=generated_caption.strip()
        )

        # Write row to CSV
        with open(out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print(f"✔  {img_path} | Actual: {species} | Predicted: {pred_label}")

print(f"All done! Running log is in →  {out_csv}")
