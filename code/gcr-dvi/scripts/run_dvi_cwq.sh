#!/bin/bash
# ============================================================
# Part 2: DVI (Decompose-Verify-Intersect) on CWQ
# Mode A: KG-specialized LLM verifier
# Mode B: v2 PathScorer verifier (pass "scorer-only" to skip Mode A)
# ============================================================

set -e
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATA_PATH=rmanluo
DATA=RoG-cwq
SPLIT=test
INDEX_LEN=2
K=10

KG_MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
KG_MODEL_NAME=$(basename "$KG_MODEL_PATH")
GENERAL_MODEL=gpt-4o-mini

if [ "${1}" != "scorer-only" ]; then
    echo "======================================================"
    echo "[DVI Mode A] KG-LLM verifier"
    echo "  KG model:      $KG_MODEL_NAME"
    echo "  General model: $GENERAL_MODEL"
    echo "  Dataset:       $DATA / $SPLIT"
    echo "======================================================"

    python workflow/predict_dvi.py \
        --data_path ${DATA_PATH} \
        --d ${DATA} \
        --split ${SPLIT} \
        --kg_model_name ${KG_MODEL_NAME} \
        --kg_model_path ${KG_MODEL_PATH} \
        --general_model_name ${GENERAL_MODEL} \
        --k ${K} \
        --index_path_length ${INDEX_LEN} \
        --attn_implementation flash_attention_2 \
        --decompose_cache_path data/decompose_cache_cwq.json \
        --min_candidates 1
fi

echo ""
echo "======================================================"
echo "[DVI Mode B] PathScorer verifier"
echo "  General model: $GENERAL_MODEL"
echo "  Dataset:       $DATA / $SPLIT"
echo "======================================================"

python workflow/predict_dvi.py \
    --data_path ${DATA_PATH} \
    --d ${DATA} \
    --split ${SPLIT} \
    --general_model_name ${GENERAL_MODEL} \
    --index_path_length ${INDEX_LEN} \
    --path_scorer \
    --bi_encoder "sentence-transformers/all-MiniLM-L6-v2" \
    --cross_encoder "cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --bi_k 100 \
    --cross_k 10 \
    --decompose_cache_path data/decompose_cache_cwq.json \
    --min_candidates 1

echo ""
echo "[DVI] Done. Results saved under: results/DVI/${DATA}/"
