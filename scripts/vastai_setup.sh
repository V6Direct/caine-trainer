#!/bin/bash
# ============================================================
#  scripts/vastai_setup.sh
#  Einmalig auf der Vast.ai Instanz ausführen nach dem Start.
#  Optimiert für Tesla A10 (24GB VRAM, Ampere, CUDA 12.x)
# ============================================================

set -e  # Bei Fehler abbrechen

echo "============================================"
echo "  Caine AI — Vast.ai Setup (Tesla A10)"
echo "============================================"

# ── 1. GPU prüfen ────────────────────────────────────────────
echo ""
echo "[1/8] GPU-Check..."
nvidia-smi
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA verfügbar: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {name}')
    print(f'VRAM: {vram:.1f} GB')
    # A10 Check
    if 'A10' not in name:
        print(f'[WARNUNG] Erwartet Tesla A10, gefunden: {name}')
        print('Config trotzdem kompatibel, aber Werte eventuell anpassen.')
"

# ── 2. System-Pakete ─────────────────────────────────────────
echo ""
echo "[2/8] System-Pakete installieren..."
apt-get update -qq
apt-get install -y -qq \
    git \
    git-lfs \
    curl \
    wget \
    screen \
    tmux \
    htop \
    nvtop \
    unzip \
    ffmpeg
git lfs install

# ── 3. Python-Abhängigkeiten ──────────────────────────────────
echo ""
echo "[3/8] Python-Pakete installieren..."
pip install --upgrade pip -q

# CUDA 12.x kompatibles PyTorch zuerst (falls nicht schon da)
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "Installiere PyTorch mit CUDA 12.1..."
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121 -q
}

pip install -r requirements.txt -q

# ── 4. Flash Attention (optional, beschleunigt Training ~20%) ─
echo ""
echo "[4/8] Flash Attention 2 installieren (kann 5 Min dauern)..."
pip install flash-attn --no-build-isolation -q && \
    echo "✅ Flash Attention installiert!" || \
    echo "⚠️  Flash Attention fehlgeschlagen — Training läuft trotzdem."

# ── 5. HuggingFace Login (für Mistral 7B) ───────────────────
echo ""
echo "[5/8] HuggingFace Login..."
echo ""
echo "⚠️  Mistral-7B-Instruct benötigt einen HuggingFace-Account"
echo "   und Zugang zum Modell (kostenlos, einmalig beantragen):"
echo "   https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
echo ""

if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN gefunden, logge ein..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "Kein HF_TOKEN gesetzt. Manuell einloggen:"
    echo "  huggingface-cli login"
    echo "  (oder HF_TOKEN als Vast.ai Environment Variable setzen)"
fi

# ── 6. WandB (optional) ──────────────────────────────────────
echo ""
echo "[6/8] WandB Setup..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "✅ WandB eingeloggt."
else
    echo "Kein WANDB_API_KEY gesetzt."
    echo "Deaktiviere WandB (nutze TensorBoard stattdessen)..."
    wandb disabled
fi

# ── 7. Anthropic API Key (für synthetische Daten) ────────────
echo ""
echo "[7/8] API Keys prüfen..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY nicht gesetzt."
    echo "   Ohne diesen Key funktioniert 'make synthetic' nicht."
    echo "   Setzen mit: export ANTHROPIC_API_KEY='sk-ant-...'"
else
    echo "✅ ANTHROPIC_API_KEY gesetzt."
fi

# ── 8. Verzeichnisse erstellen ───────────────────────────────
echo ""
echo "[8/8] Verzeichnisstruktur..."
mkdir -p data/raw data/processed checkpoints

# ── NVIDIA Einstellungen für maximale Performance ────────────
echo ""
echo "Optimiere NVIDIA-Einstellungen..."
# Persistence Mode an (verhindert GPU-Kaltstart-Latenz)
nvidia-smi -pm 1 2>/dev/null || true
# Power Limit auf Maximum (A10 = 150W default, max 150W)
nvidia-smi --power-limit=150 2>/dev/null || true

# ── tmux-Session für langes Training ─────────────────────────
echo ""
echo "============================================"
echo "  ✅ Setup abgeschlossen!"
echo "============================================"
echo ""
echo "Nächste Schritte:"
echo ""
echo "  1. Untertitel in data/raw/ ablegen (.srt oder .vtt)"
echo "     + lore.txt mit deinem Worldbuilding"
echo ""
echo "  2. Daten verarbeiten:"
echo "     make data"
echo ""
echo "  3. (Optional) Synthetische Daten generieren:"
echo "     make synthetic"
echo ""
echo "  4. Debug-Run (erst testen!):"
echo "     make train-debug"
echo ""
echo "  5. Echtes Training (in tmux/screen, weil Vast.ai Sessions abbrechen!):"
echo "     tmux new -s training"
echo "     make train"
echo "     [Ctrl+B, D] zum Detachen"
echo ""
echo "  6. Chat testen:"
echo "     make chat"
echo ""
echo "💡 Tipp: Vast.ai Instanzen können jederzeit terminiert werden."
echo "   Speichere Checkpoints regelmäßig auf deinen eigenen Storage!"
echo "   Empfehlung: rclone nach Backblaze B2 oder einen gemounteten Volume."
echo ""
