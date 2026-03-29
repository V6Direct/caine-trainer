#!/usr/bin/env python3
"""
src/train.py
QLoRA Fine-tuning von Mistral-7B-Instruct auf Caine-Dialogen.
Nutzt PEFT + TRL + BitsAndBytes für speichereffizientes Training.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import jsonlines
import wandb
from datasets import Dataset
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rich.console import Console
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("train")
console = Console()


# ─── Argumente ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Training für Caine AI")
    parser.add_argument("--config",       required=True,  help="Pfad zur train_config.yaml")
    parser.add_argument("--model_id",     default=None,   help="Überschreibt config.model.model_id")
    parser.add_argument("--output_dir",   default=None,   help="Überschreibt config.training.output_dir")
    parser.add_argument("--wandb_project",default=None)
    parser.add_argument("--max_samples",  type=int, default=None, help="Subset für Debug")
    parser.add_argument("--num_epochs",   type=int, default=None, help="Überschreibt Epochenzahl")
    parser.add_argument("--debug",        action="store_true")
    return parser.parse_args()


# ─── Daten laden ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path, max_samples: int = None) -> list[dict]:
    samples = []
    with jsonlines.open(path) as reader:
        for item in reader:
            samples.append(item)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def load_datasets(cfg, max_samples: int = None):
    data_cfg = cfg.data

    train_data = load_jsonl(Path(data_cfg.train_file), max_samples)
    eval_data  = load_jsonl(Path(data_cfg.eval_file),  max_samples // 5 if max_samples else None)

    log.info(f"Train-Samples: {len(train_data)} | Eval-Samples: {len(eval_data)}")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset  = Dataset.from_list(eval_data)

    return train_dataset, eval_dataset


# ─── Tokenisierung ────────────────────────────────────────────────────────────

def apply_chat_template(example: dict, tokenizer) -> dict:
    """
    Wendet das Mistral-Instruct Chat-Template an.
    Format: <s>[INST] {system}\n\n{user} [/INST] {assistant} </s>
    """
    messages = example["messages"]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted}


# ─── Modell laden ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg):
    model_cfg = cfg.model
    model_id  = model_cfg.model_id

    log.info(f"Lade Tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Wichtig für Mistral!

    log.info("Konfiguriere 4-bit Quantisierung (QLoRA)...")
    # Tesla A10 nutzt Ampere-Architektur (sm86) — float16 statt bfloat16!
    # bfloat16 wird auf A10 zwar unterstützt, aber fp16 ist stabiler für Training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    log.info(f"Lade Modell: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    log.info("Bereite Modell für k-bit Training vor...")
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(model, cfg):
    lora_cfg = cfg.lora
    log.info(f"Wende LoRA an (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha})...")

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        target_modules=list(lora_cfg.target_modules),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


# ─── Training Arguments ───────────────────────────────────────────────────────

def build_training_args(cfg, output_dir: str, wandb_project: str) -> TrainingArguments:
    t = cfg.training
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        gradient_checkpointing=t.gradient_checkpointing,
        optim=t.optim,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        fp16=t.fp16,
        bf16=t.bf16,
        evaluation_strategy=t.evaluation_strategy,
        eval_steps=t.eval_steps,
        logging_steps=t.logging_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        report_to="wandb" if wandb_project else "tensorboard",
        run_name=cfg.wandb.run_name if wandb_project else None,
        seed=t.seed,
        data_seed=t.data_seed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        group_by_length=True,        # Effizienter: ähnliche Längen zusammen
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Config laden und überschreiben
    cfg = OmegaConf.load(args.config)
    if args.model_id:
        cfg.model.model_id = args.model_id
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.num_epochs:
        cfg.training.num_train_epochs = args.num_epochs
    if args.debug:
        cfg.training.logging_steps = 5
        cfg.training.eval_steps = 10
        cfg.training.save_steps = 50

    output_dir  = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed setzen
    set_seed(cfg.training.seed)

    # WandB initialisieren
    wandb_project = args.wandb_project or cfg.wandb.project
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=cfg.wandb.run_name,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # GPU-Info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    else:
        log.warning("Keine GPU gefunden! Training wird sehr langsam sein.")

    # Daten laden
    train_dataset, eval_dataset = load_datasets(cfg, args.max_samples)

    # Modell + Tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)
    model, peft_config = apply_lora(model, cfg)

    # Chat-Template anwenden
    log.info("Tokenisiere Datasets...")
    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        remove_columns=eval_dataset.column_names,
    )

    # Training Args
    training_args = build_training_args(cfg, str(output_dir), wandb_project)

    # Response-only Training: Nur Caine's Antworten werden als Loss berechnet
    # Das [/INST]-Token markiert den Beginn der Antwort bei Mistral
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=cfg.training.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        packing=False,
    )

    # Training starten
    console.rule("[bold green]🎪 Training startet — Vorhang auf für Caine!")
    log.info(f"Epochen: {cfg.training.num_train_epochs} | "
             f"Batch: {cfg.training.per_device_train_batch_size} | "
             f"Grad Accum: {cfg.training.gradient_accumulation_steps}")

    trainer.train()

    # Bestes Modell speichern
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"✅ Modell gespeichert: {final_dir}")

    if wandb_project:
        wandb.finish()

    console.rule("[bold green]✅ Training abgeschlossen!")


if __name__ == "__main__":
    main()
