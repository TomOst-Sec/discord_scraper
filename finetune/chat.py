"""
Chat with your fine-tuned model.
"""
import argparse
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
LORA_PATH = "./discord-lora"
MAX_SEQ_LENGTH = 2048

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that responds in a casual, Discord-like style. "
    "Keep responses concise and conversational."
)

parser = argparse.ArgumentParser(description="Chat with a fine-tuned Discord persona model.")
parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for the model")
parser.add_argument("--lora-path", default=LORA_PATH, help="Path to LoRA adapter")
args = parser.parse_args()

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.lora_path,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

print("Bot ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit", "q"):
        break

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": user_input},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    print(f"Bot: {response}\n")
