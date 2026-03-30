#!/usr/bin/env python3
"""
src/push_to_hub.py
Uploads a local model directory (merged model or LoRA adapter) to the Hugging Face Hub.
Called by: make push

Usage:
    python src/push_to_hub.py --model_dir ./checkpoints/merged --repo_name my-caine-ai
    python src/push_to_hub.py --model_dir ./checkpoints/final  --repo_name my-caine-lora --private

Env vars:
    HF_TOKEN or HUGGINGFACE_HUB_TOKEN — your HuggingFace write token
"""

import os
import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi
from rich.logging import RichHandler
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("push_to_hub")
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--model_dir", required=True, type=Path,
        help="Local directory containing model weights (merged or LoRA adapter)"
    )
    parser.add_argument(
        "--repo_name", required=True,
        help="Target repo, e.g. 'caine-ai-v1' or 'username/caine-ai-v1'"
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create/keep the repo private (default: public)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HuggingFace write token (defaults to env HF_TOKEN)"
    )
    args = parser.parse_args()

    if not args.model_dir.exists() or not args.model_dir.is_dir():
        raise SystemExit(f"❌ Model directory not found: {args.model_dir}")

    if not args.token:
        raise SystemExit(
            "❌ No HuggingFace token found.\n"
            "   Set HF_TOKEN env var or pass --token <your_token>"
        )

    api = HfApi(token=args.token)

    # Resolve full repo_id (add username prefix if missing)
    if "/" not in args.repo_name:
        whoami = api.whoami()
        username = whoami.get("name") or whoami.get("fullname")
        repo_id = f"{username}/{args.repo_name}"
    else:
        repo_id = args.repo_name

    log.info(f"Target repo: {repo_id} (private={args.private})")

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    log.info(f"Uploading: {args.model_dir} → {repo_id}")
    api.upload_folder(
        folder_path=str(args.model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload Caine AI model from {args.model_dir.name}",
    )

    console.print(f"\n[bold green]✅ Upload complete![/bold green]")
    console.print(f"View at: [cyan]https://huggingface.co/{repo_id}[/cyan]")


if __name__ == "__main__":
    main()