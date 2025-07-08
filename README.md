# Discord Scraper

Scrape messages from any Discord user in a guild and optionally fine-tune an LLM on their conversational style.

## Setup

```bash
pip install requests
```

Set your Discord token as an environment variable:

```bash
export DISCORD_TOKEN="your_discord_token_here"
```

## Usage

### Scrape messages

```bash
python scrape.py --guild-id <GUILD_ID> --user-id <USER_ID>
```

This will:
1. Fetch all messages from the specified user in the guild
2. Save raw messages to `output/messages.jsonl`
3. Build conversation pairs (prompt/response) and save to `output/pairs.jsonl`

### Fine-tune a model (optional)

The `finetune/` directory contains scripts to fine-tune a Llama 3.1 8B model using LoRA on the scraped data.

```bash
cd finetune
bash setup.sh
```

Then chat with the fine-tuned model:

```bash
python chat.py --system-prompt "Your custom system prompt here"
```

Requires a GPU with at least 16GB VRAM (e.g., RTX 4080).

## Output

- `output/messages.jsonl` - Raw scraped messages
- `output/pairs.jsonl` - Conversation pairs (other user prompt -> target user response)
- `output/progress.json` - Scraping progress (for resuming interrupted runs)
