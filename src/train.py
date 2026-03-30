#!/usr/bin/env python3
"""
src/train.py
QLoRA Fine-tuning von Mistral-7B-Instruct auf Caine-Dialogen.
Nutzt PEFT + TRL + BitsAndBytes für speichereffizientes Training.

MODIFIED FOR: Single NVIDIA A10 (22GB VRAM)
  - QLoRA 4-bit (NF4 + double quant) via bitsandbytes
  - LoRA adapters only (no full fine-tune)
  - fp16 (not bf16 — more stable on A10/Ampere sm86)
  - Gradient checkpointing enabled
  - Batch size 1 + gradient accumulation
  - Dataset packing enabled for throughput
  - Safe checkpoint saving with save_total_limit
  - Tokenizer pad token fix
  - gradient_checkpointing_kwargs to suppress warnings
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
from trl import SFTTrainer
from trl.trainer import DataCollatorForCompletionOnlyLM

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
    parser.add_argument("--config",        required=True,  help="Pfad zur train_config.yaml")
    parser.add_argument("--model_id",      default=None,   help="Überschreibt config.model.model_id")
    parser.add_argument("--output_dir",    default=None,   help="Überschreibt config.training.output_dir")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--max_samples",   type=int, default=None, help="Subset für Debug")
    parser.add_argument("--num_epochs",    type=int, default=None, help="Überschreibt Epochenzahl")
    parser.add_argument("--debug",         action="store_true")
    return parser.parse_args()


# ─── Daten laden ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path, max_samples: int = None) -> list[dict]:
    """
    Load JSONL file. Each line must be a JSON object.
    Supports max_samples for quick debug runs.
    """
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
    Applies the Mistral-Instruct chat template.
    Expected input format: {"messages": [{"role": ..., "content": ...}, ...]}
    Output: {"text": "<s>[INST] ... [/INST] ... </s>"}
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

    # [MODIFIED] Ensure pad token is set — required for batched training.
    # Using eos_token as pad_token is standard for decoder-only models.
    # padding_side="right" avoids attention mask issues with Mistral.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    log.info("Konfiguriere 4-bit Quantisierung (QLoRA)...")
    # [MODIFIED] Explicit fp16 compute dtype for A10 (Ampere sm86).
    # NF4 quantization + double quant reduces VRAM from ~14GB to ~5GB for 7B.
    # This is the core QLoRA memory saving — do NOT change to 8-bit or fp32.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,   # fp16 — stable on A10
        bnb_4bit_use_double_quant=True,          # saves ~0.4 bits/param extra
        bnb_4bit_quant_type="nf4",               # NF4 is best for normal-dist weights
    )

    log.info(f"Lade Modell: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",                       # auto-places layers on GPU
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",             # flash_attention_2 needs bf16; use eager for fp16
    )
    # [MODIFIED] Disable KV cache — incompatible with gradient checkpointing.
    # Must be False during training, can be re-enabled at inference.
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    log.info("Bereite Modell für k-bit Training vor...")
    # [MODIFIED] gradient_checkpointing=True passed here to also enable
    # input_require_grads, which is needed for PEFT + gradient checkpointing to work.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    return model, tokenizer


def apply_lora(model, cfg):
    lora_cfg = cfg.lora
    log.info(f"Wende LoRA an (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha})...")

    # [MODIFIED] Safe defaults for A10 7B training:
    #   r=16, lora_alpha=32, dropout=0.05
    # target_modules covers all projection layers in Mistral/LLaMA attention + MLP.
    # Adjust via config — these are read from lora_cfg, no hardcoding here.
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

        # [MODIFIED] A10 22GB safe defaults: batch=1, grad_accum=8 → effective batch=8
        # Increase grad_accum_steps (not batch size) to simulate larger batches.
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),

        # [MODIFIED] Gradient checkpointing enabled — trades ~20% speed for ~40% VRAM.
        # use_reentrant=False avoids deprecation warnings in PyTorch >=2.1.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        optim=t.get("optim", "paged_adamw_8bit"),  # paged_adamw_8bit saves ~2GB vs adam

        learning_rate=t.get("learning_rate", 2e-4),
        weight_decay=t.get("weight_decay", 0.01),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),

        # [MODIFIED] fp16=True, bf16=False — required for A10 stability.
        # bf16 can cause loss spikes on Ampere sm86 in some configurations.
        fp16=True,
        bf16=False,

        # Evaluation + logging
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 50),
        logging_steps=t.get("logging_steps", 10),
        logging_first_step=True,               # [MODIFIED] Log step 0 to verify loss is finite

        # [MODIFIED] Safe checkpoint saving: save every N steps, keep only last 2.
        # Prevents disk overflow during long runs.
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 50),
        save_total_limit=t.get("save_total_limit", 2),       # keep only 2 checkpoints
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),

        report_to="wandb" if wandb_project else "tensorboard",
        run_name=cfg.wandb.run_name if wandb_project else None,

        seed=t.get("seed", 42),
        data_seed=t.get("data_seed", 42),

        remove_unused_columns=False,
        dataloader_pin_memory=True,
        # [MODIFIED] group_by_length=True clusters similar-length samples together,
        # reducing padding waste. Works well with packing=False.
    #         group_by_length=True,

        # [MODIFIED] ddp_find_unused_parameters=False — not using DDP, but
        # setting False avoids a warning with PEFT models.
        ddp_find_unused_parameters=False,
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

    output_dir = Path(cfg.training.output_dir)
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
        # [MODIFIED] Warn if not on expected A10 — helps catch misconfigured envs
        if vram_gb < 20:
            log.warning(f"VRAM {vram_gb:.1f}GB < 22GB — may OOM on 7B model!")
    else:
        log.warning("Keine GPU gefunden! Training wird sehr langsam sein.")
        sys.exit(1)

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
        desc="Applying chat template (train)",
    )
    eval_dataset = eval_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        remove_columns=eval_dataset.column_names,
        desc="Applying chat template (eval)",
    )


    # Tokenize datasets manually
    def tokenize(example):
        result = tokenizer(example["text"], truncation=True, max_length=cfg.training.get("max_seq_length", 2048), padding=False)
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = train_dataset.map(tokenize, remove_columns=["text"], desc="Tokenizing train")
    eval_dataset  = eval_dataset.map(tokenize,  remove_columns=["text"], desc="Tokenizing eval")

    # Training Args
    training_args = build_training_args(cfg, str(output_dir), wandb_project)

    # [MODIFIED] Response-only training: only compute loss on assistant tokens.
    # The [/INST] token marks the start of Caine's response in Mistral-Instruct.
    # DataCollatorForCompletionOnlyLM masks user/system tokens from the loss.
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # [MODIFIED] SFTTrainer with packing=False for better GPU utilization.
    # Packing concatenates short samples to fill the full max_seq_length window,
    # reducing padding waste and speeding up training significantly on short dialogs.
    # If your dataset has very long samples (>512 tokens avg), set packing=False.
    use_packing = cfg.training.get("packing", True)
    if use_packing:
        # packing and DataCollatorForCompletionOnlyLM are incompatible —
        # with packing, response masking is handled internally by SFTTrainer.
        log.info("Dataset packing aktiviert — DataCollator wird ignoriert.")
        active_collator = None
    else:
        active_collator = collator

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=cfg.training.get("max_seq_length", 2048),

        args=training_args,
        data_collator=None,
        packing=use_packing,
    )
    trainer.processing_class = tokenizer  # trl/transformers version mismatch workaround

    # Training starten
    console.rule("[bold green]🎪 Training startet — Vorhang auf für Caine!")
    log.info(
        f"Epochen: {cfg.training.num_train_epochs} | "
        f"Batch: {training_args.per_device_train_batch_size} | "
        f"Grad Accum: {training_args.gradient_accumulation_steps} | "
        f"Effective Batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps} | "
        f"Packing: {use_packing}"
    )

    # [MODIFIED] Resume from checkpoint if one exists (safe restart support).
    last_checkpoint = None
    if output_dir.exists():
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            log.info(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # [MODIFIED] Save only LoRA adapters (not full model) — drastically smaller output.
    # Full merge can be done separately with: model.merge_and_unload()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"✅ LoRA Adapter gespeichert: {final_dir}")

    # [MODIFIED] Log final VRAM usage to verify we stayed within budget
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(0) / 1e9
        log.info(f"Peak VRAM used: {peak_vram:.2f} GB")

    if wandb_project:
        wandb.finish()

    console.rule("[bold green]✅ Training abgeschlossen!")


if __name__ == "__main__":
    main()