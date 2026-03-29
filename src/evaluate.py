#!/usr/bin/env python3
"""
src/evaluate.py
Evaluiert das Fine-tuned Caine-Modell auf dem Test-Set.
Berechnet Perplexity, ROUGE und gibt qualitative Beispiele aus.
"""

import json
import logging
import argparse
import math
from pathlib import Path

import torch
import jsonlines
import numpy as np
from tqdm import tqdm
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("evaluate")
console = Console()


def load_model(model_dir: Path, base_model_id: str = None):
    """Lädt das Fine-tuned Modell (mit LoRA-Adapter oder gemergt)."""
    model_dir = Path(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prüfen ob LoRA-Adapter vorhanden
    adapter_config = model_dir / "adapter_config.json"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    if adapter_config.exists() and base_model_id:
        log.info(f"Lade Basismodell: {base_model_id}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        log.info("Lade LoRA-Adapter...")
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        log.info(f"Lade Modell direkt aus: {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    """Berechnet durchschnittliche Perplexity auf einem Textkorpus."""
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Perplexity"):
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(model.device)

            if enc.input_ids.shape[1] < 4:
                continue

            outputs = model(**enc, labels=enc.input_ids)
            loss = outputs.loss.item()
            if not math.isnan(loss) and not math.isinf(loss):
                total_loss += loss
                count += 1

    avg_loss = total_loss / count if count > 0 else float("inf")
    return math.exp(avg_loss)


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_message: str,
    max_new_tokens: int = 300,
) -> str:
    """Generiert eine Caine-Antwort auf eine Nutzereingabe."""
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Nur neu generierte Tokens
    new_tokens = out[0][enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Berechnet ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        scores["rouge1"].append(s["rouge1"].fmeasure)
        scores["rouge2"].append(s["rouge2"].fmeasure)
        scores["rougeL"].append(s["rougeL"].fmeasure)

    return {k: np.mean(v) for k, v in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",     required=True, type=Path)
    parser.add_argument("--test_file",     required=True, type=Path)
    parser.add_argument("--base_model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--num_examples",  type=int, default=5, help="Qualitative Beispiele")
    parser.add_argument("--max_eval",      type=int, default=100, help="Max Test-Samples")
    parser.add_argument("--system_prompt_file",
                        default="./configs/caine_system_prompt.txt", type=Path)
    args = parser.parse_args()

    # System-Prompt laden
    system_prompt = ""
    if args.system_prompt_file.exists():
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()

    # Test-Daten laden
    test_data = []
    with jsonlines.open(args.test_file) as r:
        for item in r:
            test_data.append(item)
            if len(test_data) >= args.max_eval:
                break

    log.info(f"Test-Samples: {len(test_data)}")

    # Modell laden
    model, tokenizer = load_model(args.model_dir, args.base_model_id)

    # ── Perplexity ──────────────────────────────────────────────────────────
    log.info("Berechne Perplexity...")
    texts = []
    for item in test_data:
        msgs = item.get("messages", [])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    ppl = compute_perplexity(model, tokenizer, texts)
    console.print(f"\n[bold cyan]Perplexity:[/bold cyan] {ppl:.2f}")

    # ── ROUGE ────────────────────────────────────────────────────────────────
    log.info("Berechne ROUGE-Scores...")
    predictions, references = [], []
    for item in tqdm(test_data[:50], desc="ROUGE Generation"):
        msgs = item.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        ref_response = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if not user_msg or not ref_response:
            continue

        pred = generate_response(model, tokenizer, system_prompt, user_msg)
        predictions.append(pred)
        references.append(ref_response)

    rouge_scores = compute_rouge(predictions, references)

    table = Table(title="ROUGE Scores", show_header=True)
    table.add_column("Metrik", style="bold")
    table.add_column("Score")
    for k, v in rouge_scores.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    # ── Qualitative Beispiele ────────────────────────────────────────────────
    console.rule("[bold yellow]Qualitative Beispiele")
    example_prompts = [
        "What exactly IS this place? Who are you, really?",
        "I refuse to play your games anymore.",
        "Do you actually care about any of us?",
        "What happens if we win? Does the show actually end?",
        "You're enjoying this, aren't you.",
    ]

    for prompt in example_prompts[:args.num_examples]:
        response = generate_response(model, tokenizer, system_prompt, prompt)
        console.print(f"\n[bold green]Player:[/bold green] {prompt}")
        console.print(f"[bold magenta]Caine:[/bold magenta]  {response}")
        console.print("─" * 60)

    # Ergebnisse speichern
    results = {
        "perplexity": ppl,
        "rouge": rouge_scores,
        "num_test_samples": len(test_data),
    }
    out_path = args.model_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"✅ Ergebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    main()
