import requests
import json
import time
import os
import argparse

TOKEN = os.environ.get("DISCORD_TOKEN")
if not TOKEN:
    raise RuntimeError("Set the DISCORD_TOKEN environment variable before running.")

HEADERS = {
    "Authorization": TOKEN,
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/145.0.0.0 Safari/537.36",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def api_request(url, max_retries=5):
    """Make an API request with retry logic for transient errors."""
    for attempt in range(max_retries):
        r = requests.get(url, headers=HEADERS)

        if r.status_code == 429:
            retry_after = r.json().get("retry_after", 5)
            print(f"  Rate limited, waiting {retry_after}s...")
            time.sleep(retry_after)
            continue

        if r.status_code in (500, 502, 503, 504):
            wait = 5 * (attempt + 1)
            print(f"  Server error {r.status_code}, retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
            continue

        return r

    return r


def load_progress():
    """Load previously saved progress."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        print(f"  Resuming from {len(data['messages'])} previously fetched messages...")
        return data["messages"], set(data["seen_ids"]), data.get("max_id")
    return [], set(), None


def save_progress(all_msgs, seen_ids, max_id):
    """Save progress incrementally."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "messages": all_msgs,
            "seen_ids": list(seen_ids),
            "max_id": max_id,
        }, f)


def search_user_messages(guild_id, user_id):
    """Use Discord's search API to fetch messages from a specific user.
    Uses max_id pagination to bypass the 10k offset limit.
    Saves progress incrementally."""
    all_msgs, seen_ids, max_id = load_progress()
    total = None

    print(f"Searching for all messages from user {user_id} in guild {guild_id}...")

    while True:
        offset = 0

        while True:
            url = (
                f"https://discord.com/api/v9/guilds/{guild_id}/messages/search"
                f"?author_id={user_id}&offset={offset}"
            )
            if max_id:
                url += f"&max_id={max_id}"

            r = api_request(url)

            if r.status_code == 400:
                break

            r.raise_for_status()
            data = r.json()

            if total is None:
                total = data.get("total_results", 0)
                print(f"  Found {total} total messages from this user.")

            if not data.get("messages"):
                break

            batch_count = 0
            for msg_group in data["messages"]:
                for m in msg_group:
                    if m.get("hit") and m["id"] not in seen_ids:
                        all_msgs.append(m)
                        seen_ids.add(m["id"])
                        batch_count += 1

            if batch_count == 0:
                break

