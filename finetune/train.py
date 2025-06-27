"""
LoRA fine-tune using unsloth on Llama 3.1 8B.
Optimized for RTX 4080 16GB VRAM.
"""
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# === Config ===
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
TRAIN_DATA = "train_data.jsonl"
OUTPUT_DIR = "./discord-lora"

# Load model with 4-bit quantization (fits in 16GB VRAM)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load training data
dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")


def formatting_func(examples):
    """Format conversations into chat template."""
    texts = []
    for convos in examples["conversations"]:
        text = tokenizer.apply_chat_template(convos, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


dataset = dataset.map(formatting_func, batched=True)

# Training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        optim="adamw_8bit",
    ),
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=True,
)

print("Starting training...")
trainer.train()

# Save the LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone! LoRA adapter saved to {OUTPUT_DIR}")
print("Use chat.py to test it out.")
