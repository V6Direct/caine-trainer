#!/bin/bash
# ============================================================
#  scripts/backup_checkpoints.sh
#  Sichert Checkpoints automatisch — wichtig auf Vast.ai,
#  weil Instanzen jederzeit terminiert werden können!
#
#  Nutzt rclone (unterstützt Backblaze B2, S3, Google Drive, etc.)
#  Setup: rclone config → Remote "backup" einrichten
# ============================================================

CHECKPOINT_DIR="./checkpoints"
REMOTE="backup:caine-ai-checkpoints"   # rclone Remote:Pfad
LOG_FILE="./backup.log"
INTERVAL=1800  # Alle 30 Minuten sichern

echo "[$(date)] Backup-Script gestartet (alle ${INTERVAL}s)" | tee -a "$LOG_FILE"

while true; do
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo "[$(date)] Starte Backup..." | tee -a "$LOG_FILE"
        rclone sync "$CHECKPOINT_DIR" "$REMOTE" \
            --progress \
            --transfers=4 \
            --log-file="$LOG_FILE" \
            --log-level=INFO
        echo "[$(date)] Backup abgeschlossen." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] Checkpoint-Dir nicht gefunden: $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
    fi
    sleep "$INTERVAL"
done
