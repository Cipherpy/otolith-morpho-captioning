# import requests
# import torch
# from PIL import Image
# from transformers import MllamaForConditionalGeneration, AutoProcessor
# with open("/home/reshma/Otolith/captioning/otolith/prompts/user_prompt.txt", "r") as f:
#     user_prompt = f.read().strip()  # use actual prompt
# model_id = "meta-llama/Llama-3.2-11B-Vision"

# model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# processor = AutoProcessor.from_pretrained(model_id)

# url = "/home/reshma/Otolith/captioning/otolith/model_data/test/Dactyloptena orientalis/Dactyloptena orientalis_Sp10_R (9).tif"
# image = Image.open(url)

# prompt = user_prompt
# inputs = processor(image, prompt, return_tensors="pt").to(model.device)

# output = model.generate(**inputs, max_new_tokens=30)
# print(processor.decode(output[0]))


from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from PIL import Image

with open("/home/reshma/Otolith/captioning/otolith/prompts/user_prompt.txt", "r") as f:
    user_prompt = f.read().strip()  # use actual prompt

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

FastVisionModel.for_inference(model) # Enable for inference!

image = Image.open("/home/reshma/Otolith/captioning/otolith/model_data/test/Minous inermis/AD54R 1.6X.tif")


messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": user_prompt}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = False, temperature = 1.5, min_p = 0.1)

print(_)
