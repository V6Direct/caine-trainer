#!/usr/bin/env python3
"""
src/merge_adapter.py
Merged den trainierten LoRA-Adapter mit dem Basismodell
zu einem vollständigen, standalone Modell (kein PEFT mehr nötig).

MODIFIED: default dtype changed from bfloat16 to float16
          to match A10 training setup.
"""

import argparse
import logging
from pathlib import Path

import torch
from rich.logging import RichHandler
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("merge_adapter")
console = Console()


def main():
    parser = argparse.ArgumentParser(description="LoRA-Adapter mit Basismodell mergen")
    parser.add_argument("--base_model",  required=True, help="HF Model ID oder lokaler Pfad")
    parser.add_argument("--adapter_dir", required=True, type=Path)
    parser.add_argument("--output_dir",  required=True, type=Path)
    # [MODIFIED] default changed from bfloat16 to float16 — matches A10 training dtype
    parser.add_argument("--dtype",       default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="float16 recommended for A10-trained adapters")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    log.info(f"Lade Basismodell: {args.base_model} ({args.dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    log.info(f"Lade LoRA-Adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        torch_dtype=torch_dtype,
    )

    log.info("Merge läuft (kann einige Minuten dauern)...")
    model = model.merge_and_unload()
    model.eval()

    log.info(f"Speichere gemergtes Modell → {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    log.info("Speichere Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    console.print(f"\n[bold green]✅ Merge abgeschlossen![/bold green]")
    console.print(f"Modell gespeichert unter: [cyan]{args.output_dir}[/cyan]")
    console.print(
        "\n[dim]Du kannst das Modell jetzt ohne PEFT laden:\n"
        "  AutoModelForCausalLM.from_pretrained(output_dir)[/dim]"
    )


if __name__ == "__main__":
    main()