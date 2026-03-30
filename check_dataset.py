import json
from pathlib import Path
from collections import Counter

# adjust path if needed
path = Path("data/tadc_dataset.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

print(f"TOTAL ROWS: {len(rows)}")

print("\n=== FIRST 3 ROWS ===")
for i, row in enumerate(rows[:3], 1):
    print(f"\n--- ROW {i} ---")
    print(json.dumps(row, indent=2, ensure_ascii=False))

print("\n=== TOP LEVEL KEYS ===")
keys = Counter()
for row in rows:
    keys.update(row.keys())
print(keys)

print("\n=== MESSAGE ROLES ===")
roles = Counter()
for row in rows:
    for msg in row.get("messages", []):
        roles[msg.get("role", "MISSING")] += 1
print(roles)

print("\n=== SYSTEM PROMPTS (first 3) ===")
count = 0
for row in rows:
    for msg in row.get("messages", []):
        if msg.get("role") == "system":
            print(msg["content"][:200])
            print("---")
            count += 1
            if count >= 3:
                break
    if count >= 3:
        break