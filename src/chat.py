#!/usr/bin/env python3
"""
src/chat.py
Interaktiver Terminal-Chat mit dem Fine-tuned Caine-Modell.
Zum lokalen Testen nach dem Training.
"""

import argparse
import logging
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel

logging.basicConfig(level=logging.WARNING)
console = Console()


def load_model(model_dir: Path, base_model_id: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists() and base_model_id:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


def generate(
    model, tokenizer, messages: list[dict],
    temperature: float = 0.85,
    top_p: float = 0.92,
    max_new_tokens: int = 400,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    new_tokens = out[0][enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",         required=True, type=Path)
    parser.add_argument("--base_model_id",     default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--system_prompt",     default="./configs/caine_system_prompt.txt")
    parser.add_argument("--temperature",       type=float, default=0.85)
    parser.add_argument("--top_p",             type=float, default=0.92)
    parser.add_argument("--max_new_tokens",    type=int,   default=400)
    parser.add_argument("--no_memory",         action="store_true",
                        help="Kein Gesprächsverlauf (jeder Turn unabhängig)")
    args = parser.parse_args()

    # System-Prompt laden
    sp_path = Path(args.system_prompt)
    system_prompt = sp_path.read_text(encoding="utf-8").strip() if sp_path.exists() else \
        "You are Caine, theatrical host of a surreal game show."

    console.print(Panel.fit(
        "[bold magenta]🎪 Caine AI — Interaktiver Chat[/bold magenta]\n"
        "[dim]Tippe 'exit' oder 'quit' zum Beenden[/dim]\n"
        "[dim]Tippe 'reset' um den Gesprächsverlauf zu löschen[/dim]\n"
        "[dim]Tippe 'temp X' um Temperatur zu ändern (z.B. 'temp 0.9')[/dim]",
        border_style="magenta",
    ))

    console.print("[dim]Lade Modell...[/dim]")
    model, tokenizer = load_model(args.model_dir, args.base_model_id)
    console.print("[green]✅ Modell geladen![/green]\n")

    # Gesprächsverlauf
    history: list[dict] = []
    temperature = args.temperature

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]Du[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Auf Wiedersehen![/dim]")
            break

        if not user_input:
            continue

        # Befehle
        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]Vorhang fällt. Auf Wiedersehen![/dim]")
            break

        if user_input.lower() == "reset":
            history = []
            console.print("[yellow]Gesprächsverlauf gelöscht.[/yellow]")
            continue

        if user_input.lower().startswith("temp "):
            try:
                temperature = float(user_input.split()[1])
                console.print(f"[yellow]Temperatur: {temperature}[/yellow]")
            except (IndexError, ValueError):
                console.print("[red]Ungültige Temperatur.[/red]")
            continue

        # Nachricht hinzufügen
        history.append({"role": "user", "content": user_input})

        # Messages für diesen Turn bauen
        messages = [{"role": "system", "content": system_prompt}]
        if args.no_memory:
            messages.append({"role": "user", "content": user_input})
        else:
            messages.extend(history)

        console.print("\n[bold magenta]Caine[/bold magenta]: ", end="")
        response = generate(
            model, tokenizer, messages,
            temperature=temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        console.print()  # Newline nach Stream

        if not args.no_memory:
            history.append({"role": "assistant", "content": response})

        console.print()


if __name__ == "__main__":
    main()
