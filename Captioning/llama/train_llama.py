# Set up environment
import torch
from unsloth import FastVisionModel
from datasets import load_dataset
import os
import pandas as pd
from PIL import Image
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb logging

# Load Unsloth's 4-bit quantized Llama 3.2 Vision model
print("Loading Llama 3.2 Vision 11B model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

print("Model loaded successfully!")
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Enable training mode
FastVisionModel.for_training(model)

# Configure LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # Train vision components
    finetune_language_layers=True,    # Train language components  
    finetune_attention_modules=True,  # Train attention layers
    finetune_mlp_modules=True,        # Train MLP layers
    r=16,                            # LoRA rank
    lora_alpha=16,                   # LoRA scaling parameter
    lora_dropout=0,                  # No dropout for stability
    bias="none",                     # No bias training
    random_state=3407,               # For reproducibility
    use_rslora=False,                # Standard LoRA
    loftq_config=None,               # No quantization-aware training
)

print("Model configured for fine-tuning!")


with open("/home/reshma/Otolith/captioning/otolith/prompts/user_prompt.txt", "r") as f:
    user_prompt = f.read()


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
    # Create conversation format that Llama expects
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": f"{caption}\nClass detected:{class_name}"}]
        }
    ]
    return {
        "messages": conversation,
        "image": image
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


train_csv_path = "/home/reshma/Otolith/captioning/otolith/text_data/train_modified.csv"
eval_csv_path = "/home/reshma/Otolith/captioning/otolith/text_data/val_modified.csv"

train_base_dir = "/home/reshma/Otolith/captioning/otolith/model_data/train"
eval_base_dir = "/home/reshma/Otolith/captioning/otolith/model_data/val"


train_dataset = load_dataset_from_csv(train_csv_path, train_base_dir)
eval_dataset=load_dataset_from_csv(eval_csv_path, eval_base_dir)


from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 30,
        num_train_epochs = 50, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "/gpu_das_hdd/reshma/llama/",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        #max_seq_length = 2048,
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")