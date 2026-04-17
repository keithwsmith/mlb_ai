# finetune.py
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

# Load base model (4-bit quantized to fit in less VRAM)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name  = "defog/sqlcoder-7b-2",
    max_seq_length = 2048,
    dtype       = None,       # auto-detect
    load_in_4bit = True,      # quantize to 4-bit
)

# Add LoRA adapters — only trains a small % of weights
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,       # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = True,
)

# Load your data
with open("training_data.json") as f:
    raw = json.load(f)

# Format into prompt template sqlcoder expects
def format_prompt(example):
    return {
        "text": f"""### Instructions:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }

dataset = Dataset.from_list(raw).map(format_prompt)

# Train
trainer = SFTTrainer(
    model        = model,
    tokenizer    = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps   = 5,
        num_train_epochs = 10,     # more epochs = more memorization
        learning_rate  = 2e-4,
        fp16           = True,
        logging_steps  = 1,
        output_dir     = "./sqlcoder_mlb_finetuned",
        save_strategy  = "epoch",
    ),
)

trainer.train()

# Save the LoRA adapter (small file, ~50MB)
model.save_pretrained("sqlcoder_mlb_lora")
tokenizer.save_pretrained("sqlcoder_mlb_lora")
print("Fine-tune complete. Adapter saved to sqlcoder_mlb_lora/")