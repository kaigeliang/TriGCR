#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
DATA_PATH=${DATA_PATH:-rmanluo}
DATA=${DATA:-RoG-cwq}
INDEX_LEN=${INDEX_LEN:-2}
MAX_NEG=${MAX_NEG:-50}
OUT_DIR=${OUT_DIR:-data/path_verifier/RoG-cwq}

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" workflow/build_path_verifier_data.py \
  --data_path "${DATA_PATH}" \
  --d "${DATA}" \
  --split "${TRAIN_SPLIT:-train[:2000]}" \
  --index_path_length "${INDEX_LEN}" \
  --max_negatives_per_question "${MAX_NEG}" \
  --negative_strategy hard \
  --output "${OUT_DIR}/train.jsonl" \
  --grouped_output "${OUT_DIR}/train.grouped.jsonl"

"${PYTHON_BIN}" workflow/build_path_verifier_data.py \
  --data_path "${DATA_PATH}" \
  --d "${DATA}" \
  --split "${DEV_SPLIT:-train[2000:2500]}" \
  --index_path_length "${INDEX_LEN}" \
  --max_negatives_per_question "${MAX_NEG}" \
  --negative_strategy hard \
  --output "${OUT_DIR}/dev.jsonl" \
  --grouped_output "${OUT_DIR}/dev.grouped.jsonl"
