#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
DATA_DIR=${DATA_DIR:-data/path_verifier/RoG-cwq}
MODEL_NAME=${MODEL_NAME:-cross-encoder/ms-marco-MiniLM-L-6-v2}
OUTPUT_DIR=${OUTPUT_DIR:-results/path_reranker/RoG-cwq/msmarco-minilm}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-1}
DEVICE=${DEVICE:-cuda}

"${PYTHON_BIN}" workflow/train_path_reranker.py \
  --train_file "${DATA_DIR}/train.jsonl" \
  --dev_file "${DATA_DIR}/dev.jsonl" \
  --grouped_dev_file "${DATA_DIR}/dev.grouped.jsonl" \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --use_amp
