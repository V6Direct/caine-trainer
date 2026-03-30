import json
import random
from pathlib import Path
from collections import Counter

path = Path("data/tadc_dataset.jsonl")
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def get_character(row):
    for msg in row.get("messages", []):
        if msg["role"] == "system":
            content = msg["content"]
            if content.startswith("You are ") and "from The Amazing Digital Circus" in content:
                return content.replace("You are ", "").replace(" from The Amazing Digital Circus.", "").strip()
    return "UNKNOWN"

# Show distribution
char_counts = Counter(get_character(r) for r in rows)
print("\n=== CHARACTER DISTRIBUTION ===")
for char, count in char_counts.most_common():
    print(f"  {char}: {count} samples")

# Buckets — everything not Caine or Bubble goes to NPCs
CAINE_CHARS  = {"Caine"}
BUBBLE_CHARS = {"Bubble"}

buckets = {"caine": [], "bubble": [], "npcs": []}
for row in rows:
    char = get_character(row)
    if char in CAINE_CHARS:
        buckets["caine"].append(row)
    elif char in BUBBLE_CHARS:
        buckets["bubble"].append(row)
    else:
        buckets["npcs"].append(row)

print("\n=== BUCKET SIZES ===")
for name, items in buckets.items():
    print(f"  {name}: {len(items)} samples")

def split(data, train=0.85, ev=0.08):
    random.seed(42)
    random.shuffle(data)
    n = len(data)
    t = int(n * train)
    v = int(n * (train + ev))
    return data[:t], data[t:v], data[v:]

out_base = Path("data/characters")
out_base.mkdir(parents=True, exist_ok=True)

for name, items in buckets.items():
    if len(items) == 0:
        print(f"WARNING: No samples for {name}! Consider generating synthetic data.")
        continue
    train_set, eval_set, test_set = split(items)
    for split_name, split_data in [("train", train_set), ("eval", eval_set), ("test", test_set)]:
        out_path = out_base / f"{name}_{split_name}.jsonl"
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in split_data), encoding="utf-8")
        print(f"  Saved {out_path}: {len(split_data)} samples")

print("\n✅ Split complete!")