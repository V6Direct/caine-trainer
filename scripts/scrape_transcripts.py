#!/usr/bin/env python3
"""
TADC Transcript Scraper (Fandom API version - stable)
"""

import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# أين output folder
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://tadc.fandom.com/api.php"

EPISODES = {
    "pilot":         "THE_AMAZING_DIGITAL_CIRCUS:_PILOT/Transcript",
    "ep2_candy":     "Candy_Carrier_Chaos!/Transcript",
    "ep3_manor":     "The_Mystery_Of_Mildenhall_Manor/Transcript",
    "ep4_fastfood":  "Fast_Food_Masquerade/Transcript",
    "ep5_untitled":  "Untitled/Transcript",
    "ep6_guns":      "They_All_Get_Guns/Transcript",
    "ep7_beach":     "Beach_Episode/Transcript",
    "ep8_hjsakldfhl":"Hjsakldfhl/Transcript",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def fetch_page(title: str):
    params = {
        "action": "parse",
        "page": title,
        "format": "json"
    }

    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "parse" not in data:
            print("  ⚠️ API returned no parse data (page may still exist)")
            return None

        return data["parse"]["text"]["*"]

    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        return None


def extract_lines(html: str):
    soup = BeautifulSoup(html, "html.parser")
    lines = []

    # scan paragraphs + list items
    for tag in soup.find_all(["p", "li"]):
        text = tag.get_text(strip=True)

        if ":" in text:
            speaker, dialogue = text.split(":", 1)

            speaker = speaker.strip()
            dialogue = dialogue.strip()

            # basic sanity filters
            if (
                len(speaker) < 30
                and len(dialogue) > 3
                and not speaker.lower().startswith(("note", "trivia"))
            ):
                lines.append(f"{speaker}: {dialogue}")

    return lines


def scrape_episode(name: str, page_title: str):
    print(f"Scraping {name}...")

    html = fetch_page(page_title)
    if not html:
        print("  ⚠️ Skipped (no HTML)")
        return 0

    lines = extract_lines(html)

    if not lines:
        print("  ⚠️ No dialogue found")
        return 0

    out_path = OUTPUT_DIR / f"{name}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  ✅ {len(lines)} lines → {out_path}")
    return len(lines)


def main():
    print("=" * 50)
    print("  TADC Transcript Scraper (API version)")
    print("=" * 50)

    total = 0

    for name, title in EPISODES.items():
        count = scrape_episode(name, title)
        total += count
        time.sleep(1)  # be nice to API

    print()
    print(f"✅ Done! {total} lines total in data/raw/")


if __name__ == "__main__":
    main()