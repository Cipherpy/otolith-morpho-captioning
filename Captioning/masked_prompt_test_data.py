import torch, os, pandas as pd, csv
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

system_message = "You are an expert in marine taxonomy, especially marine fishes."
# After loading base prompt from file:
with open("/home/reshma/Otolith/captioning/otolith/prompts/user_prompt.txt", "r") as f:
    base_user_prompt = f.read().strip()

# Add strict “see-only” rules while preserving your tag format
STRICT_VISUAL_POLICY = """
FOLLOW THESE RULES EXACTLY:

GENERAL
- Describe ONLY what is clearly visible in THIS image. Do not rely on memory, prior images, dataset patterns, file names, or any external knowledge.
- If a requested feature is occluded, cropped, blurred, too small, or uncertain, write exactly: "not visible".
- Do NOT guess or infer unseen details. Avoid hedging words like "likely", "probably", "appears to".
- Use precise taxonomic vocabulary when visible; otherwise use "not visible".
- Keep statements short, factual, and tied to what the pixels show.

FORMAT
- Keep the SAME tags, order, and structure as provided below. Do NOT add or remove tags.
- For each tag, provide 1–2 concise sentences; no bullet lists unless the template explicitly calls for them.
- Do NOT add any text before the first tag or after the last tag.

SPECIES PREDICTION
- Provide a scientific name ONLY if the visible morphology is sufficient and unambiguous.
- If not sufficient, write exactly: "Unknown" for the species name, then list key observed features and what additional visible features would be needed to decide.
- Do NOT cite training data, prior experience, or match to previously seen images.
"""

# This becomes the prompt used in generation
user_prompt = base_user_prompt + "\n\n" + STRICT_VISUAL_POLICY

def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                image = element.get("image", element)
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# def generate_description(image, model, processor):
#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": system_message}]},
#         {"role": "user", "content": [
#             {"type": "text", "text": user_prompt},
#             {"type": "image","image": image},
#         ]},
#     ]
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to(model.device)
#     stop_token_ids = [
#         processor.tokenizer.eos_token_id,
#         processor.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
#     ]
#     generated_ids = model.generate(
#         **inputs,
#         max_new_tokens=256,
#         top_p=1.0,
#         do_sample=True,
#         temperature=0.8,
#         eos_token_id=stop_token_ids,
#         disable_compile=True
#     )
#     generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     return output_text[0]



def generate_description(sample, model, processor):
    # Convert sample into messages and then apply the chat template
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image","image": sample["image"]},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process the image and text
    image_inputs = process_vision_info(messages)
    # Tokenize the text and process the images
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move the inputs to the device
    inputs = inputs.to(model.device)
    
    # Generate the output
    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8, eos_token_id=stop_token_ids, disable_compile=True)
    # Trim the generation and decode the output to text
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def extract_predicted_label(caption):
    
    # Try to extract label from the generated caption, e.g., look for "Class detected: ..."
    import re
    m = re.search(r"Class(?: detected)?:\s*(.+)", caption)

   
    if m:
        return m.group(1)
    # Fallback: look for species or class in text (customize as needed)
    return ""

# === Main script ===

test_base_dir = "/home/reshma/Otolith/captioning/otolith/masked_data"  # <-- your test data path

# Hugging Face model id
model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`


# Load Model base model
model = AutoModelForImageTextToText.from_pretrained(model_id, low_cpu_mem_usage=True)

output_dir="/gpu_das_hdd/reshma/gemma-otolith"

#Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("/gpu_das_hdd/reshma/gemma-otolith/merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoProcessor.from_pretrained(output_dir)
processor.save_pretrained("/gpu_das_hdd/reshma/gemma-otolith/merged_model")

#Load model/processor
model = AutoModelForImageTextToText.from_pretrained("/gpu_das_hdd/reshma/gemma-otolith/merged_model")
processor = AutoProcessor.from_pretrained("/gpu_das_hdd/reshma/gemma-otolith/merged_model")



import torch

# Load Model with PEFT adapter
model = AutoModelForImageTextToText.from_pretrained(
  output_dir,
  device_map="auto",
  torch_dtype=torch.bfloat16,
  attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(output_dir)

# results = []

# for species in sorted(os.listdir(test_base_dir)):
#     species_dir = os.path.join(test_base_dir, species)
#     if not os.path.isdir(species_dir):
#         continue
#     for img_file in sorted(os.listdir(species_dir)):
#         if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
#             continue
#         img_path = os.path.join(species_dir, img_file)
#         try:
#             sample = {
#                 "image": Image.open(img_path).convert("RGB")
#             }
#             generated_caption = generate_description(sample, model, processor)
#             print(generated_caption)
#             predicted_label = extract_predicted_label(generated_caption)
#             results.append({
#                 "image_path": img_path,
#                 "actual_label": species,
#                 "predicted_label": predicted_label,
#                 "generated_caption": generated_caption
#             })
#             print(f"Processed {img_path} | Actual: {species} | Predicted: {predicted_label}")
#         except Exception as e:
#             print(f"Failed {img_path}: {e}")

# # Save to Excel
# df = pd.DataFrame(results)
# df.to_excel("otolith_test_results.xlsx", index=False)
# print("Results saved to otolith_test_results.xlsx")


########################################################################
# 1)  OUTPUT FILE SET-UP
########################################################################
out_csv = "prompt_masked_otolith_test_results_progress.csv"
fieldnames = ["image_path", "actual_label",
              "predicted_label", "generated_caption"]

#  ➜  Create the file with a header once, or resume if it already exists.
if not os.path.isfile(out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

########################################################################
# 2)  MAIN LOOP  (unchanged until the very end of the inner for-loop)
########################################################################
for species in sorted(os.listdir(test_base_dir)):
    species_dir = os.path.join(test_base_dir, species)
    if not os.path.isdir(species_dir):
        continue

    for img_file in sorted(os.listdir(species_dir)):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        img_path = os.path.join(species_dir, img_file)

        try:
            sample = {"image": Image.open(img_path).convert("RGB")}
            caption = generate_description(sample, model, processor)
            pred_label = extract_predicted_label(caption)

            row = dict(
                image_path   = img_path,
                actual_label = species,
                predicted_label = pred_label,
                generated_caption = caption
            )

            # --- NEW: append the row immediately -----------------------
            with open(out_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            # -----------------------------------------------------------

            print(f"✔  {img_path} | Actual: {species} | Predicted: {pred_label}")

        except Exception as e:
            print(f"✖  {img_path}: {e}")

print(f"All done! Running log is in →  {out_csv}")