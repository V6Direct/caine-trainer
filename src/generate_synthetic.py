#!/usr/bin/env python3
"""
src/generate_synthetic.py
Generiert synthetische Caine-Dialoge basierend auf Lore-Dokumenten.
Nutzt die Anthropic API als "Teacher-Modell" um Trainingsdaten zu erzeugen.

MODIFIED: Claude model ID updated to claude-haiku-4-5 (valid Oct 2025+)
"""

import os
import json
import time
import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import jsonlines
from tqdm import tqdm
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("generate_synthetic")


# ─── Szenarien für die Datengenerierung ───────────────────────────────────────

SCENARIO_TEMPLATES = [
    "Explain the rules of a new, absurd game you've just invented for the contestants.",
    "Announce the start of a challenge in your most theatrical manner.",
    "React to a contestant unexpectedly succeeding at something you thought was impossible.",
    "React to a contestant dramatically failing at the simplest task.",
    "Introduce a sudden rule change mid-game with a smile.",
    "A contestant demands to know who you really are and why you're doing this.",
    "A contestant refuses to participate. What do you say?",
    "Two contestants are arguing with each other. Step in as the host.",
    "A contestant tries to break the rules. Respond in character.",
    "A contestant asks if any of this is real.",
    "A contestant begs you to let them go home.",
    "A contestant asks if you actually care about them.",
    "A contestant says they trust you. Respond authentically as Caine.",
    "Narrate the beginning of a new episode to an unseen audience.",
    "Announce the elimination of a contestant dramatically.",
    "Give a speech about the purpose of the show.",
    "React to the show going completely off-script in the best possible way.",
    "React to something unexpected happening that you did NOT plan.",
    "Describe what happens when someone wins — or does anyone ever truly win?",
    "Muse philosophically about the nature of games and reality.",
    "Describe a new area of the world the contestants have just entered.",
    "Explain an unusual power or ability you are demonstrating.",
    "Answer a question about how the world works without revealing too much.",
    "React to a contestant finding a loophole you didn't anticipate.",
    "Describe your favorite game you've ever hosted and why.",
    "A contestant accuses you of being cruel. Respond.",
    "Justify why the game must continue despite someone getting hurt.",
    "Deliver bad news to the contestants with your signature style.",
    "Explain what happens to contestants who are eliminated.",
    "React when the show itself seems to be breaking down around you.",
]

CONTESTANT_NAMES = [
    "Pomni", "Jax", "Ragatha", "Gangle", "Kinger", "Zooble",
    "Alex", "Quinn", "Riley", "Morgan", "Jordan", "Casey",
]

SYSTEM_PROMPT_FOR_GENERATOR = """You are a creative writing assistant helping to generate training data for an AI character called Caine.

Caine is the omnipotent, theatrical host of a surreal game show set inside a digital world. He is:
- Dramatically enthusiastic, performative, and loves spectacle
- Morally ambiguous — not evil, but the show always comes first
- Warm toward contestants in a detached, calculated way
- Never breaks character; reframes everything as part of the show
- Speaks with formal, slightly archaic flair and rhetorical flourish
- Improvises freely and delights in surprises and twists
- Speaks to an unseen audience as much as to the contestants

Generate ONE dialogue exchange in JSON format with these exact fields:
{
  "scenario": "brief description of the situation",
  "contestant_name": "name of the contestant speaking",
  "user_message": "what the contestant says (1-3 sentences)",
  "caine_response": "Caine's full response (3-8 sentences, in character)"
}

Rules for Caine's response:
- Stay deeply in character at all times
- Use theatrical language and performative energy
- Never give a plain, boring answer
- Let personality shine through every line
- Responses should feel alive, surprising, and creative
- ONLY return the JSON object, nothing else."""


def load_lore(lore_file: Path) -> str:
    if not lore_file.exists():
        log.warning(f"Lore-Datei nicht gefunden: {lore_file}")
        return ""
    return lore_file.read_text(encoding="utf-8").strip()


def generate_sample_via_api(
    scenario: str,
    lore_context: str,
    contestant_name: str,
    api_key: str,
    retries: int = 3,
) -> Optional[dict]:
    import httpx

    prompt = f"""Lore context for this world:
{lore_context[:1500] if lore_context else 'A surreal digital game show world.'}

Scenario: {scenario}
Contestant name: {contestant_name}

Generate a realistic, in-character dialogue exchange for this scenario."""

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        # [MODIFIED] Corrected model ID — claude-haiku-4-5 is the valid API name
        # See: https://platform.claude.com/docs/about-claude/models/overview
        "model": "claude-haiku-4-5",
        "max_tokens": 600,
        "system": SYSTEM_PROMPT_FOR_GENERATOR,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()
            raw = data["content"][0]["text"].strip()

            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            required_keys = {"scenario", "contestant_name", "user_message", "caine_response"}
            if not required_keys.issubset(parsed.keys()):
                log.warning(f"Unvollständiges JSON (Versuch {attempt+1}): {parsed.keys()}")
                continue

            return parsed

        except json.JSONDecodeError as e:
            log.warning(f"JSON Parse-Fehler (Versuch {attempt+1}): {e}")
            time.sleep(1.5)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning(f"Rate limit erreicht. Warte {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"HTTP Error: {e}")
                break
        except Exception as e:
            log.error(f"Unbekannter Fehler (Versuch {attempt+1}): {e}")
            time.sleep(2)

    return None


def sample_to_mistral_format(sample: dict, system_prompt: str) -> dict:
    context = f"[Setting: {sample.get('scenario', '')}]"
    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": f"{context}\n\n{sample['user_message']}"},
            {"role": "assistant", "content": sample["caine_response"]},
        ],
        "source": "synthetic",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generiert synthetische Caine-Trainingsdialoge via Anthropic API"
    )
    parser.add_argument("--lore_file",   required=True,  type=Path)
    parser.add_argument("--output",      required=True,  type=Path)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--api_key",     default=os.environ.get("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--delay",       type=float, default=0.5,
                        help="Sekunden zwischen API-Calls (Rate-Limiting)")
    args = parser.parse_args()

    if not args.api_key:
        log.error(
            "Kein API Key gefunden! Setze ANTHROPIC_API_KEY als Umgebungsvariable "
            "oder übergib --api_key."
        )
        return

    lore = load_lore(args.lore_file)
    log.info(f"Lore geladen: {len(lore)} Zeichen")

    sp_path = Path("./configs/caine_system_prompt.txt")
    system_prompt = sp_path.read_text(encoding="utf-8").strip() if sp_path.exists() else ""

    args.output.parent.mkdir(parents=True, exist_ok=True)

    generated = []
    failed = 0

    scenarios = SCENARIO_TEMPLATES.copy()
    random.shuffle(scenarios)
    extended = (scenarios * ((args.num_samples // len(scenarios)) + 2))[:args.num_samples]

    with jsonlines.open(args.output, mode="w") as writer:
        for i, scenario in enumerate(tqdm(extended, desc="Generiere Samples")):
            contestant = random.choice(CONTESTANT_NAMES)
            sample = generate_sample_via_api(
                scenario=scenario,
                lore_context=lore,
                contestant_name=contestant,
                api_key=args.api_key,
            )

            if sample is None:
                failed += 1
                continue

            formatted = sample_to_mistral_format(sample, system_prompt)
            writer.write(formatted)
            generated.append(formatted)

            if args.delay > 0:
                time.sleep(args.delay)

            if (i + 1) % 50 == 0:
                log.info(f"Fortschritt: {len(generated)} generiert, {failed} fehlgeschlagen")

    log.info(
        f"✅ Fertig! {len(generated)} Samples gespeichert → {args.output} "
        f"({failed} fehlgeschlagen)"
    )


if __name__ == "__main__":
    main()