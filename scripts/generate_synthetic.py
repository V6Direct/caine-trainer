#!/usr/bin/env python3
import json
import random
from pathlib import Path
import openai  # oder ollama/groq

openai.api_key = "your-key-here"  # Oder lokal mit ollama

parser.add_argument("--character",           default="Caine",  help="Character name to generate for")
parser.add_argument("--system_prompt_file",  default="./configs/caine_system_prompt.txt")

def load_caine_prompt():
    return Path("configs/caine_system_prompt.txt").read_text(encoding='utf-8')

def generate_synthetic_samples(base_dataset_path, count=1000, model="gpt-4o-mini"):
    """Generiert synthetic Caine-Dialoge - robust für verschiedene Formate."""
    with open(base_dataset_path, 'r', encoding='utf-8') as f:
        base_data = [json.loads(line) for line in f]
    
    synthetic = []
    caine_prompt = load_caine_prompt()
    
    for i in range(count):
        base_example = random.choice(base_data)
        
        # Robust: Finde User-Input egal welches Format
        user_content = ""
        if "messages" in base_example:
            for msg in base_example["messages"]:
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
        elif "input" in base_example:
            user_content = base_example["input"]
        elif "instruction" in base_example:
            user_content = base_example["instruction"]
        else:
            user_content = str(base_example)[:200]  # Fallback
        
        if not user_content:
            continue
            
        # GPT Call (ersetze mit deiner API oder lokal ollama)
        try:
            response = openai.ChatCompletion.create(  # ← Dein API-Key nötig!
                model=model,
                messages=[
                    {"role": "system", "content": caine_prompt},
                    {"role": "user", "content": f"Als Caine antworten: {user_content}"}
                ],
                max_tokens=150,
                temperature=0.8
            )
            caine_response = response.choices[0].message.content.strip()
        except:
            # Fallback: Paraphrase original
            caine_response = base_example.get("output", "Ah, welch wunderbares Abenteuer!")
        
        synthetic.append({
            "messages": [  # ChatML Format für VLM
                {"role": "system", "content": "Du bist Caine."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": caine_response}
            ]
        })
        
        if i % 100 == 0:
            print(f"Generated {i}/{count}")
    
    Path("data/synthetic").mkdir(exist_ok=True)
    output_path = "data/synthetic/synthetic_multimodal.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in synthetic:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ {count} synthetic samples → {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    generate_synthetic_samples("data/tadc_dataset.jsonl", args.count)