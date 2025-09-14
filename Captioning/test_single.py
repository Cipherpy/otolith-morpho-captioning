import torch, os, pandas as pd, csv
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

system_message = "You are an expert in marine taxonomy, especially marine fishes."
with open("/home/reshma/Otolith/captioning/otolith/prompts/user_prompt.txt", "r") as f:
    user_prompt = f.read().strip()  # use actual prompt

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
            {"type": "image","image": sample["image"]},
            {"type": "text", "text": user_prompt}
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
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=192,
        min_new_tokens=32,        # avoid extremely short generic blurbs
        do_sample=True,
        temperature=0.4,          # lower = more anchored
        top_p=0.9,                # nucleus sampling
        top_k=50,                 # cap sampling pool further
        repetition_penalty=1.05,  # discourages rote phrasing & species-name priors
        eos_token_id=stop_token_ids,
        disable_compile=True)
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

test_image = "/home/reshma/Otolith/captioning/otolith/model_data/test/Setarches guentheri/AH46 0.75X.tif"  # <-- your test data path

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



# Load Model with PEFT adapter
model = AutoModelForImageTextToText.from_pretrained(
  output_dir,
  device_map="auto",
  torch_dtype=torch.bfloat16,
  attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(output_dir)



sample = {"image": Image.open(test_image).convert("RGB")}
caption = generate_description(sample, model, processor)
pred_label = extract_predicted_label(caption)

print(caption)