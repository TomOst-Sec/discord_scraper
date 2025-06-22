#!/bin/bash
set -e

echo "Installing unsloth + dependencies..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets

echo ""
echo "Preparing training data..."
python prepare_data.py

echo ""
echo "Starting fine-tune..."
python train.py

echo ""
echo "Done! Run 'python chat.py' to test your model."
