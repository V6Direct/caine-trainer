# ============================================================
#  Caine AI — Training Pipeline Makefile
#  Basismodell: Mistral-7B-Instruct-v0.3
#  Methode:     QLoRA (4-bit) via PEFT + TRL
# ============================================================

PYTHON      := python3
PIP         := pip3
MODEL_ID    := mistralai/Mistral-7B-Instruct-v0.3
OUTPUT_DIR  := ./checkpoints
DATA_DIR    := ./data
CONFIG      := ./configs/train_config.yaml
WANDB_PROJ  := caine-ai

.PHONY: all setup install clean data synthetic train eval merge push help

all: setup data synthetic train eval

# -----------------------------------------------------------
# 1. Environment Setup
# -----------------------------------------------------------
setup:
	@echo "🎪 Setting up Caine AI environment..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m nltk.downloader punkt stopwords
	@echo "✅ Environment ready."

install: setup

# -----------------------------------------------------------
# 2. Datenverarbeitung
# -----------------------------------------------------------
data:
	@echo "📄 Verarbeite Rohdaten..."
	$(PYTHON) src/process_subtitles.py \
		--input_dir $(DATA_DIR)/raw \
		--output_dir $(DATA_DIR)/processed \
		--character "Caine"
	@echo "✅ Daten verarbeitet."

# -----------------------------------------------------------
# 3. Synthetische Daten generieren (Caine-Persönlichkeit)
# -----------------------------------------------------------
synthetic:
	@echo "🤖 Generiere synthetische Trainingsdialoge..."
	$(PYTHON) src/generate_synthetic.py \
		--lore_file $(DATA_DIR)/raw/lore.txt \
		--output $(DATA_DIR)/processed/synthetic.jsonl \
		--num_samples 500
	@echo "✅ Synthetische Daten generiert."

# -----------------------------------------------------------
# 4. Training (QLoRA Fine-tuning)
# -----------------------------------------------------------
train:
	@echo "🎭 Starte QLoRA Training..."
	$(PYTHON) src/train.py \
		--config $(CONFIG) \
		--model_id $(MODEL_ID) \
		--output_dir $(OUTPUT_DIR) \
		--wandb_project $(WANDB_PROJ)
	@echo "✅ Training abgeschlossen."

# Schneller Test-Run (1 Epoch, kleines Subset)
train-debug:
	@echo "🔍 Debug-Training (1 Epoch, 50 Samples)..."
	$(PYTHON) src/train.py \
		--config $(CONFIG) \
		--model_id $(MODEL_ID) \
		--output_dir $(OUTPUT_DIR)/debug \
		--wandb_project $(WANDB_PROJ) \
		--max_samples 50 \
		--num_epochs 1 \
		--debug

# -----------------------------------------------------------
# 5. Evaluation
# -----------------------------------------------------------
eval:
	@echo "📊 Evaluiere Modell..."
	$(PYTHON) src/evaluate.py \
		--model_dir $(OUTPUT_DIR)/final \
		--test_file $(DATA_DIR)/processed/test.jsonl
	@echo "✅ Evaluation fertig."

# Interaktiver Chat zum Testen
chat:
	@echo "💬 Starte Chat mit Caine..."
	$(PYTHON) src/chat.py \
		--model_dir $(OUTPUT_DIR)/final \
		--system_prompt configs/caine_system_prompt.txt

# -----------------------------------------------------------
# 6. Adapter mit Basismodell mergen
# -----------------------------------------------------------
merge:
	@echo "🔧 Merge LoRA-Adapter mit Basismodell..."
	$(PYTHON) src/merge_adapter.py \
		--base_model $(MODEL_ID) \
		--adapter_dir $(OUTPUT_DIR)/final \
		--output_dir $(OUTPUT_DIR)/merged
	@echo "✅ Merge abgeschlossen. Modell unter $(OUTPUT_DIR)/merged"

# Auf HuggingFace pushen
push:
	@echo "🚀 Push zu HuggingFace Hub..."
	$(PYTHON) src/push_to_hub.py \
		--model_dir $(OUTPUT_DIR)/merged \
		--repo_name caine-ai-v1
	@echo "✅ Modell hochgeladen."

# -----------------------------------------------------------
# Cleanup
# -----------------------------------------------------------
clean:
	@echo "🧹 Cleanup..."
	rm -rf $(OUTPUT_DIR)/debug
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned."

clean-all: clean
	rm -rf $(OUTPUT_DIR)
	@echo "⚠️  Alle Checkpoints gelöscht."

# -----------------------------------------------------------
# Vast.ai spezifisch
# -----------------------------------------------------------
vastai-setup:
	@echo "🚀 Vast.ai Setup für Tesla A10..."
	@chmod +x scripts/vastai_setup.sh
	@bash scripts/vastai_setup.sh

backup:
	@echo "💾 Starte automatisches Checkpoint-Backup..."
	@chmod +x scripts/backup_checkpoints.sh
	@bash scripts/backup_checkpoints.sh

	# -----------------------------------------------------------
# Per-character training targets
# -----------------------------------------------------------
split:
	@echo "✂️  Splitting dataset by character..."
	$(PYTHON) scripts/split_by_character.py
	@echo "✅ Split complete."

train-caine:
	@echo "🎪 Training Caine..."
	$(PYTHON) src/train.py --config configs/caine.yaml --wandb_project $(WANDB_PROJ)

train-bubble:
	@echo "🫧 Training Bubble..."
	$(PYTHON) src/train.py --config configs/bubble.yaml --wandb_project $(WANDB_PROJ)

train-npcs:
	@echo "🎭 Training NPCs..."
	$(PYTHON) src/train.py --config configs/npcs.yaml --wandb_project $(WANDB_PROJ)

train-all: train-caine train-bubble train-npcs
	@echo "✅ All characters trained!"

chat-caine:
	$(PYTHON) src/chat.py --model_dir ./checkpoints/caine/final --system_prompt configs/caine_system_prompt.txt

chat-bubble:
	$(PYTHON) src/chat.py --model_dir ./checkpoints/bubble/final --system_prompt configs/bubble_system_prompt.txt

debug-caine:
	$(PYTHON) src/train.py --config configs/caine.yaml --max_samples 50 --num_epochs 1 --debug

debug-bubble:
	$(PYTHON) src/train.py --config configs/bubble.yaml --max_samples 30 --num_epochs 1 --debug

debug-npcs:
	$(PYTHON) src/train.py --config configs/npcs.yaml --max_samples 50 --num_epochs 1 --debug


	synthetic-caine:
	@echo "🎪 Generating synthetic Caine data..."
	$(PYTHON) src/generate_synthetic.py \
		--lore_file $(DATA_DIR)/raw/lore.txt \
		--output $(DATA_DIR)/synthetic/caine_synthetic.jsonl \
		--num_samples 300 \
		--character "Caine" \
		--system_prompt_file configs/caine_system_prompt.txt

synthetic-bubble:
	@echo "🫧 Generating synthetic Bubble data..."
	$(PYTHON) src/generate_synthetic.py \
		--lore_file $(DATA_DIR)/raw/lore.txt \
		--output $(DATA_DIR)/synthetic/bubble_synthetic.jsonl \
		--num_samples 400 \
		--character "Bubble" \
		--system_prompt_file configs/bubble_system_prompt.txt
help:
	@echo ""
	@echo "  Caine AI - Training Pipeline"
	@echo "  ================================"
	@echo "  make setup       – Abhängigkeiten installieren"
	@echo "  make data        – Untertitel & Rohdaten verarbeiten"
	@echo "  make synthetic   – Synthetische Dialoge generieren"
	@echo "  make train       – QLoRA Training starten"
	@echo "  make train-debug – Schneller Test-Run"
	@echo "  make eval        – Modell evaluieren"
	@echo "  make chat        – Interaktiver Chat"
	@echo "  make merge       – LoRA-Adapter mergen"
	@echo "  make push        – Auf HuggingFace pushen"
	@echo "  make vastai-setup – Vast.ai Instanz einrichten (A10)"
	@echo "  make backup       – Checkpoints automatisch sichern"
	@echo "  make clean        – Temp-Dateien löschen"
	@echo ""
