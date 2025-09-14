import torch
from transformers import pipeline
import pandas as pd
import os
from PIL import Image  
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig 
import requests
from PIL import Image


system_message = "You are an expert in marine taxonomy, especially marine fishes."

with open("/home/reshma/Otolith/captioning/otolith/prompts/user_prompt.txt", "r") as f:
    user_prompt = "Describe the morphology of this otolith specimen, taxonomically based on the features used by taxonomists."

def load_dataset_from_csv(csv_path, img_base_dir=None):
    df = pd.read_csv(csv_path)
    
    dataset = []
    for _, row in df.iterrows():
        marine_class = row["Class"]
        image_file = row["Image"]

        class_name = marine_class.split(".")[0]

        img_path = f"{img_base_dir}/{marine_class}/{image_file}"
        
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue
        
        try:
            image = Image.open(img_path)
            formatted_sample = format_data(image, row['Description'], class_name)
            dataset.append(formatted_sample)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    return dataset

def format_data(image, caption, class_name):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"{caption}\nClass detected:{class_name}"}],
            },
        ],
    }

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
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs




# Hugging Face model id
model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained("/home/reshma/Otolith/captioning/otolith/merged_model")
processor = AutoProcessor.from_pretrained("/home/reshma/Otolith/captioning/otolith/merged_model")



# Test sample with Product Name, Category and Image
sample = {
  "image": Image.open("/home/reshma/Otolith/captioning/Image003.tif").convert("RGB")
}

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

# generate the description
description = generate_description(sample, model, processor)
print(description)