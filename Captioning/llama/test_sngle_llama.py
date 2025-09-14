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
image_path = "/home/reshma/Otolith/captioning/otolith/model_data/test/Setarches guentheri/AH46 0.75X.tif"

# --- Helper to extract predicted class from caption ---
import re
def extract_predicted_label(caption):
    m = re.search(r"Class(?: detected)?:\s*(.+)", str(caption))
    if m:
        return m.group(1).strip()
    return ""

# --- Main loop over subfolders and images ---



try:
    image = Image.open(image_path).convert('RGB')
except Exception as e:
    print(f"✖  {image_path}: {e}")

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
            temperature=0.4,
            min_p=0.1
        )
        generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(generated_caption)
except Exception as e:
    print(f"✖  Error generating caption for {image_path}: {e}")
    generated_caption = ""

# Extract predicted label from generated caption
pred_label = extract_predicted_label(generated_caption)



print(f"✔  {image_path} | Actual: {species} | Predicted: {pred_label}")
