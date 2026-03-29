# 🎪 Caine AI — Training Pipeline

Fine-tuning von Mistral-7B-Instruct auf Caine-Charakter-Dialogen via QLoRA.

---

## Projektstruktur

```
caine-ai/
├── Makefile
├── requirements.txt
├── configs/
│   ├── train_config.yaml        ← Alle Hyperparameter
│   └── caine_system_prompt.txt  ← Caines Persönlichkeit
├── data/
│   ├── raw/                     ← Hier kommen deine Rohdaten rein
│   │   ├── episode01.srt        ← YouTube-Untertitel (.srt oder .vtt)
│   │   ├── episode02.vtt
│   │   └── lore.txt             ← Dein Lore-Dokument
│   └── processed/               ← Wird automatisch befüllt
│       ├── train.jsonl
│       ├── eval.jsonl
│       └── test.jsonl
├── src/
│   ├── process_subtitles.py     ← Untertitel → Trainingsdaten
│   ├── generate_synthetic.py    ← KI-generierte Zusatzdaten
│   ├── train.py                 ← QLoRA Fine-tuning
│   ├── evaluate.py              ← Perplexity + ROUGE + Beispiele
│   ├── chat.py                  ← Interaktiver Test-Chat
│   └── merge_adapter.py         ← LoRA in Basismodell mergen
└── checkpoints/                 ← Wird automatisch erstellt
    ├── checkpoint-200/
    ├── checkpoint-400/
    └── final/
```

---

## Schritt-für-Schritt Workflow

### 1. RunPod-Instanz aufsetzen

Empfohlenes Template auf RunPod:
- **GPU**: NVIDIA A40 (48GB VRAM) oder A100 (80GB) — A40 reicht locker
- **Image**: `runpod/pytorch:2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04`
- **Disk**: mindestens 80GB (Basismodell ~15GB, Checkpoints ~20GB)

```bash
# Im RunPod Terminal:
git clone https://github.com/DEIN-REPO/caine-ai.git
cd caine-ai
make setup
```

### 2. Rohdaten vorbereiten

Lege deine Dateien in `data/raw/` ab:

```
data/raw/
├── episode01.srt     ← YouTube-Untertitel herunterladen (yt-dlp --write-subs)
├── episode02.srt
├── ...
└── lore.txt          ← Dein Worldbuilding-Dokument (Freitext, je mehr desto besser)
```

**Wichtig für Untertitel:** Die Sprechernamen müssen im Format `CAINE: Text` stehen.
YouTube-Untertitel haben das meist NICHT automatisch — du musst sie manuell taggen
oder ein separates Speaker-Diarization-Tool nutzen (z.B. WhisperX).

**yt-dlp Untertitel herunterladen:**
```bash
pip install yt-dlp
yt-dlp --write-subs --sub-lang en --skip-download "https://youtube.com/watch?v=..."
```

### 3. Daten verarbeiten

```bash
make data
```

Extrahiert alle Caine-Dialoge und erstellt Train/Eval/Test-Splits.

### 4. Synthetische Daten generieren (optional aber empfohlen)

Generiert 500 zusätzliche Dialoge via Anthropic API (braucht API-Key):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
make synthetic
```

Kostet ca. 2–4 USD für 500 Samples mit claude-haiku.

### 5. Training starten

```bash
export WANDB_API_KEY="..."   # optional, für Monitoring
make train
```

**Erwartete Trainingszeit auf A40:**
- 1000 Samples, 4 Epochen → ~45–90 Minuten
- 5000 Samples, 4 Epochen → ~4–6 Stunden

### 6. Debug-Run (erst testen!)

```bash
make train-debug
```

Läuft 1 Epoche auf 50 Samples — fertig in ~5 Minuten. Damit prüfst du ob alles funktioniert.

### 7. Modell testen

```bash
make chat
```

Startet interaktiven Terminal-Chat mit deinem Caine.

### 8. Evaluation

```bash
make eval
```

Gibt Perplexity, ROUGE-Scores und qualitative Beispielantworten aus.

### 9. Adapter mergen (für Production)

```bash
make merge
```

Merged LoRA-Gewichte in das Basismodell → standalone Modell ohne PEFT-Abhängigkeit.

---

## Untertitel-Format

Damit `process_subtitles.py` die Charakternamen erkennt, müssen die
Untertitel-Textzeilen so aussehen:

```
CAINE: Oh, magnificent! Truly, truly spectacular.
POMNI: What... what even is this place?
CAINE: Why, it's your new home, of course!
```

oder mit eckigen Klammern:
```
[Caine] Oh, magnificent! Truly, truly spectacular.
```

Falls deine Untertitel keine Sprechernamen haben, musst du sie manuell
hinzufügen oder WhisperX mit Speaker Diarization nutzen:

```bash
pip install whisperx
whisperx audio.mp3 --diarize --hf_token HF_TOKEN
```

---

## Hyperparameter anpassen

Alles in `configs/train_config.yaml`:

| Parameter | Default | Wann anpassen? |
|-----------|---------|----------------|
| `lora.r` | 64 | Höher (128) bei mehr Daten / mehr VRAM |
| `training.learning_rate` | 2e-4 | Runter bei Overfitting |
| `training.num_train_epochs` | 4 | Mehr bei kleinem Dataset |
| `training.per_device_train_batch_size` | 2 | Runter bei OOM-Fehler |
| `training.gradient_accumulation_steps` | 8 | Hoch bei kleiner Batch-Size |

---

## Typische Fehler

**CUDA Out of Memory:**
```yaml
# In train_config.yaml:
training:
  per_device_train_batch_size: 1   # von 2 auf 1
  gradient_accumulation_steps: 16  # von 8 auf 16 (effektive Batch-Size gleich)
```

**Keine Samples extrahiert:**
- Stelle sicher dass Sprechernamen in Untertiteln vorhanden sind
- Format muss `CAINE: Text` oder `[Caine] Text` sein

**WandB-Fehler:**
```bash
wandb disabled   # WandB deaktivieren wenn kein Account
```

**Langsames Training:**
- Flash Attention 2 installieren: `pip install flash-attn --no-build-isolation`
- In `train.py`: `attn_implementation="flash_attention_2"`

---

## Daten-Qualität ist entscheidend

Mehr als Hyperparameter bestimmt die Qualität deiner Trainingsdaten das Ergebnis:

1. **Wenige gute Samples > viele schlechte Samples**
2. **Untertitel manuell prüfen** — YouTube-Untertitel haben oft Fehler
3. **Lore-Dokument so detailliert wie möglich** — je mehr Kontext, desto besser die synthetischen Daten
4. **Mindestens 200 echte Caine-Zeilen** für gute Ergebnisse
5. **Synthetische Daten** ergänzen, nicht ersetzen

---

## Lizenz

Nur für private, nicht-kommerzielle Nutzung. Trainiere keine Modelle auf
urheberrechtlich geschütztem Material und stelle sie nicht öffentlich zur Verfügung.
