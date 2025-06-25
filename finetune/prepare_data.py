"""
Prepare scraped Discord messages into training format for unsloth LoRA fine-tuning.
Uses conversation pairs (prompt -> completion) as the primary training data.
"""
import json
import argparse

INPUT_PAIRS = "../output/pairs.jsonl"
INPUT_RAW = "../output/messages.jsonl"
OUTPUT_FILE = "train_data.jsonl"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that responds in a casual, Discord-like style. "
    "Keep responses concise and conversational."
)


def main():
    parser = argparse.ArgumentParser(description="Prepare Discord messages for fine-tuning.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for training data")
    parser.add_argument("--pairs", default=INPUT_PAIRS, help="Path to conversation pairs JSONL")
    parser.add_argument("--raw", default=INPUT_RAW, help="Path to raw messages JSONL")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output training data file")
    args = parser.parse_args()

    training_data = []

    # Load conversation pairs (best training signal)
    with open(args.pairs) as f:
        for line in f:
            pair = json.loads(line)
            training_data.append({
                "conversations": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["completion"]},
                ]
            })

    # Also include raw messages as single-turn examples
    with open(args.raw) as f:
        raw_msgs = [json.loads(line) for line in f]

    # Get pair completions to avoid duplicates
    pair_completions = set()
    with open(args.pairs) as f:
        for line in f:
            pair_completions.add(json.loads(line)["completion"])

    standalone_count = 0
    for msg in raw_msgs:
        text = msg["text"].strip()
        if text and text not in pair_completions and len(text) > 10:
            training_data.append({
                "conversations": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": "what do you think?"},
                    {"role": "assistant", "content": text},
                ]
            })
            standalone_count += 1

    with open(args.output, "w") as f:
        for item in training_data:
            json.dump(item, f)
            f.write("\n")

    print(f"Conversation pairs: {len(training_data) - standalone_count}")
    print(f"Standalone messages: {standalone_count}")
    print(f"Total training examples: {len(training_data)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
