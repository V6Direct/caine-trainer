#!/usr/bin/env python3
"""
src/process_subtitles.py
Verarbeitet YouTube-Untertitel (.srt/.vtt) und extrahiert
Caine-Dialoge als Trainingspaare im Mistral-Chat-Format.
"""

import re
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import jsonlines
from tqdm import tqdm
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("process_subtitles")


# ─── Datenstrukturen ──────────────────────────────────────────────────────────

@dataclass
class DialogueTurn:
    speaker: str
    text: str
    timestamp: Optional[str] = None


@dataclass
class TrainingSample:
    instruction: str          # Was der User / Contestant sagt
    response: str             # Was Caine antwortet
    context: str = ""         # Optionaler Kontext (vorherige Szene)
    source: str = "subtitles"


# ─── Plain Text Parser (Character-Grouped Format) ────────────────────────────

def parse_grouped_text(filepath: Path) -> list[DialogueTurn]:
    """
    Parst das Format:
        Caine:
        Line 1
        Line 2
        ...
        Kinger:
        Line 1
        ...
    """
    content = filepath.read_text(encoding="utf-8", errors="replace")
    turns = []
    current_speaker = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Sprechername-Zeile erkennen: endet mit ":" und hat keine Leerzeichen
        # (oder nur ein Wort + Doppelpunkt)
        speaker_match = re.match(r"^([A-Za-z][A-Za-z\s]{0,30}):$", line)
        if speaker_match:
            current_speaker = speaker_match.group(1).strip()
            continue

        # Normale Line dem aktuellen Sprecher zuordnen
        if current_speaker and line:
            turns.append(DialogueTurn(
                speaker=current_speaker,
                text=line,
            ))

    return turns


# ─── SRT Parser ──────────────────────────────────────────────────────────────

def parse_srt(filepath: Path) -> list[DialogueTurn]:
    """Parst eine .srt Datei und gibt eine Liste von Turns zurück."""
    content = filepath.read_text(encoding="utf-8", errors="replace")
    turns = []

    # SRT-Block Regex: Index → Timestamp → Text
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        timestamp_line = lines[1] if "-->" in lines[1] else None
        text_lines = lines[2:] if timestamp_line else lines[1:]
        raw_text = " ".join(text_lines).strip()

        # HTML-Tags entfernen (<i>, <b>, etc.)
        raw_text = re.sub(r"<[^>]+>", "", raw_text).strip()

        if not raw_text:
            continue

        # Speaker-Tag erkennen: "CAINE: ..." oder "[Caine] ..."
        speaker_match = re.match(
            r"^(?:\[)?([A-Z][a-zA-Z\s]+?)(?:\])?:\s*(.+)$", raw_text
        )
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            text = speaker_match.group(2).strip()
        else:
            speaker = "UNKNOWN"
            text = raw_text

        turns.append(DialogueTurn(
            speaker=speaker,
            text=text,
            timestamp=timestamp_line,
        ))

    return turns


def parse_vtt(filepath: Path) -> list[DialogueTurn]:
    """Parst eine .vtt Datei (WebVTT-Format)."""
    content = filepath.read_text(encoding="utf-8", errors="replace")
    turns = []

    # WEBVTT Header überspringen
    content = re.sub(r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        # Skip cue identifier lines (reine Nummern oder IDs)
        start = 0
        if lines and not "-->" in lines[0]:
            start = 1
        if len(lines) <= start:
            continue

        text_lines = [l for l in lines[start:] if "-->" not in l]
        raw_text = " ".join(text_lines).strip()
        raw_text = re.sub(r"<[^>]+>", "", raw_text).strip()

        if not raw_text:
            continue

        speaker_match = re.match(
            r"^(?:\[)?([A-Z][a-zA-Z\s]+?)(?:\])?:\s*(.+)$", raw_text
        )
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            text = speaker_match.group(2).strip()
        else:
            speaker = "UNKNOWN"
            text = raw_text

        turns.append(DialogueTurn(speaker=speaker, text=text))

    return turns


# ─── Dialog-Extraktion ────────────────────────────────────────────────────────

def extract_caine_pairs(
    turns: list[DialogueTurn],
    character: str = "Caine",
    context_window: int = 2,
) -> list[TrainingSample]:
    """
    Extrahiert Trainingspaare:
    - instruction = Was jemand VOR Caine sagt
    - response    = Was Caine darauf antwortet
    - context     = Die letzten N Turns als Kontext
    """
    samples = []
    char_upper = character.upper()

    for i, turn in enumerate(turns):
        if turn.speaker.upper() != char_upper:
            continue

        # Suche nach dem letzten Nicht-Caine-Turn davor
        prev_turn = None
        for j in range(i - 1, -1, -1):
            if turns[j].speaker.upper() != char_upper:
                prev_turn = turns[j]
                break

        if prev_turn is None:
            # Kein Gesprächspartner gefunden → Caine spricht solo (Monolog)
            # Trotzdem als Instruction → Response formatieren
            instruction = "What do you have to say?"
        else:
            instruction = prev_turn.text

        # Kontext: letzten N Turns vor diesem
        context_turns = turns[max(0, i - context_window) : i]
        context = " | ".join(
            f"{t.speaker}: {t.text}" for t in context_turns
            if t.speaker.upper() != char_upper
        )

        samples.append(TrainingSample(
            instruction=instruction,
            response=turn.text,
            context=context,
        ))

    return samples


# ─── Mistral Chat Format ──────────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path("./configs/caine_system_prompt.txt")

def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return "You are Caine, the theatrical host of a surreal game show."


def to_mistral_format(sample: TrainingSample, system_prompt: str) -> dict:
    """
    Konvertiert ein TrainingSample in das Mistral-Instruct Chat-Format:
    <s>[INST] {system}\n\n{context}\n{user} [/INST] {assistant} </s>
    """
    user_content = sample.instruction
    if sample.context:
        user_content = f"[Scene context: {sample.context}]\n\n{sample.instruction}"

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": sample.response},
        ],
        "source": sample.source,
    }


# ─── Split Helper ─────────────────────────────────────────────────────────────

def train_eval_test_split(
    samples: list,
    train: float = 0.85,
    ev: float = 0.10,
) -> tuple[list, list, list]:
    import random
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    t = int(n * train)
    v = int(n * (train + ev))
    return samples[:t], samples[t:v], samples[v:]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verarbeitet Untertitel zu Caine-Trainingsdaten"
    )
    parser.add_argument("--input_dir",  required=True, help="Ordner mit .srt/.vtt Dateien")
    parser.add_argument("--output_dir", required=True, help="Ausgabeordner für .jsonl")
    parser.add_argument("--character",  default="Caine", help="Name des Charakters")
    parser.add_argument("--context_window", type=int, default=2)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt()

    # Alle Untertiteldateien einlesen
    all_turns: list[DialogueTurn] = []
    subtitle_files = (
        list(input_dir.glob("*.srt")) +
        list(input_dir.glob("*.vtt")) +
        list(input_dir.glob("*.txt"))
    )

    if not subtitle_files:
        log.warning(f"Keine .srt/.vtt/.txt Dateien in {input_dir} gefunden!")
        return

    log.info(f"Gefunden: {len(subtitle_files)} Dateien")

    for f in tqdm(subtitle_files, desc="Parsing"):
        if f.suffix == ".srt":
            turns = parse_srt(f)
        elif f.suffix == ".vtt":
            turns = parse_vtt(f)
        else:
            # .txt → character-grouped Format
            turns = parse_grouped_text(f)
            log.info(f"  {f.name}: {len(turns)} Lines (grouped format)")
        all_turns.extend(turns)

    log.info(f"Gesamt Turns: {len(all_turns)}")

    # Caine-Paare extrahieren
    samples = extract_caine_pairs(all_turns, args.character, args.context_window)
    log.info(f"Extrahierte Caine-Paare: {len(samples)}")

    if not samples:
        log.error(
            f"Keine Samples gefunden! Stelle sicher, dass der Charakter "
            f"'{args.character}' in den Untertiteln mit seinem Namen markiert ist. "
            f"Format: 'CAINE: Text...' oder '[Caine] Text...'"
        )
        return

    # Mistral-Format anwenden
    formatted = [to_mistral_format(s, system_prompt) for s in samples]

    # Split
    train_set, eval_set, test_set = train_eval_test_split(formatted)
    log.info(f"Split → Train: {len(train_set)} | Eval: {len(eval_set)} | Test: {len(test_set)}")

    # Speichern
    for split_name, split_data in [
        ("train", train_set),
        ("eval",  eval_set),
        ("test",  test_set),
    ]:
        out_path = output_dir / f"{split_name}.jsonl"
        with jsonlines.open(out_path, mode="w") as writer:
            writer.write_all(split_data)
        log.info(f"Gespeichert: {out_path} ({len(split_data)} Samples)")

    log.info("✅ Datenverarbeitung abgeschlossen!")


if __name__ == "__main__":
    main()
