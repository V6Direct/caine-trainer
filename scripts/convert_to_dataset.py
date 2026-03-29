#!/usr/bin/env python3
"""
Convert raw transcript lines into AI training dataset (context-aware)
"""

import json
from pathlib import Path

INPUT_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/tadc_dataset.jsonl")

# normalize names
NAME_MAP = {
    "CAINE": "Caine",
    "Caine (voice)": "Caine",
    "POMNI": "Pomni",
}

WINDOW = 3  # number of previous lines as context


def normalize(name: str):
    name = name.strip()
    return NAME_MAP.get(name, name)


def parse_line(line: str):
    if ":" not in line:
        return None, None
    speaker, text = line.split(":", 1)
    return normalize(speaker), text.strip()


def main():
    conversations = []

    for file in INPUT_DIR.glob("*.txt"):
        lines = file.read_text(encoding="utf-8").splitlines()

        parsed = []
        for line in lines:
            speaker, text = parse_line(line)
            if speaker and text:
                parsed.append((speaker, text))

        # 🔥 context-based conversations
        for i in range(WINDOW, len(parsed)):
            context = parsed[i - WINDOW:i]
            speaker, reply = parsed[i]

            # skip if same speaker appears in context (avoids weird loops)
            if any(speaker == s for s, _ in context):
                continue

            context_text = "\n".join([f"{s}: {t}" for s, t in context])

            convo = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are {speaker} from The Amazing Digital Circus."
                    },
                    {
                        "role": "user",
                        "content": context_text
                    },
                    {
                        "role": "assistant",
                        "content": reply
                    }
                ]
            }

            conversations.append(convo)

    # save JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for convo in conversations:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")

    print(f"✅ Done! {len(conversations)} training samples → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()