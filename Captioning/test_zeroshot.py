import torch
from transformers import pipeline

with open("/home/reshma/Otolith/captioning/otolith/prompts/user_prompt.txt", "r") as f:
    user_prompt = f.read().strip()  # use actual prompt
pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it", # "google/gemma-3-12b-it", "google/gemma-3-27b-it" 
    device="cuda",
    torch_dtype=torch.bfloat16
)

messages = [
    {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert in marine taxonomy, especially marine fishes."}]
        },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/reshma/Otolith/captioning/otolith/model_data/test/Dactyloptena orientalis/Dactyloptena orientalis_Sp10_R (9).tif"},
            {"type": "text", "text": user_prompt}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])