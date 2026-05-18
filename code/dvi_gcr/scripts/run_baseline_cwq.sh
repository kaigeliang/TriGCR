#!/bin/bash
# ============================================================
# Part 1: Reproduce GCR Baseline on CWQ
# 测试两个预训练模型: Llama-3.1-8B 和 Qwen2-0.5B
# 满足 "Explore more than 2 base LLMs" (10%) 要求
# ============================================================

set -e

DATA_PATH=rmanluo
DATA=RoG-cwq
SPLIT=test
INDEX_LEN=2
K=10
ATTN_IMP=flash_attention_2
GENERAL_MODEL=gpt-4o-mini

# ── 模型列表 (可直接从 HuggingFace 自动下载) ──────────────────
MODELS=(
    "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"
    "rmanluo/GCR-Qwen2-0.5B-Instruct"
    # "rmanluo/GCR-Llama-2-7b-chat-hf"   # 可选第三个
)

# ── Step 1: Graph-Constrained Decoding ────────────────────────
# KG-specialized LLM 生成推理路径
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo ""
    echo "======================================================"
    echo "[Step 1] Decoding with: $MODEL_NAME"
    echo "======================================================"

    python workflow/predict_paths_and_answers.py \
        --data_path ${DATA_PATH} \
        --d ${DATA} \
        --split ${SPLIT} \
        --index_path_length ${INDEX_LEN} \
        --model_name ${MODEL_NAME} \
        --model_path ${MODEL_PATH} \
        --k ${K} \
        --prompt_mode zero-shot \
        --generation_mode group-beam \
        --attn_implementation ${ATTN_IMP}

    echo "[Step 1] Done. Paths saved to: results/GenPaths/${DATA}/${MODEL_NAME}/${SPLIT}/"
done

# ── Step 2: Graph Inductive Reasoning ─────────────────────────
# General LLM 基于路径生成最终答案
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    REASONING_PATH="results/GenPaths/${DATA}/${MODEL_NAME}/${SPLIT}/zero-shot-group-beam-k${K}-index_len${INDEX_LEN}/predictions.jsonl"

    echo ""
    echo "======================================================"
    echo "[Step 2] Final answer with $GENERAL_MODEL for: $MODEL_NAME"
    echo "======================================================"

    python workflow/predict_final_answer.py \
        --data_path ${DATA_PATH} \
        --d ${DATA} \
        --split ${SPLIT} \
        --model_name ${GENERAL_MODEL} \
        --reasoning_path ${REASONING_PATH} \
        --add_path True \
        -n 8

    echo "[Step 2] Done."
done

echo ""
echo "======================================================"
echo "All baseline runs complete!"
echo "Results saved under: results/FinalAnswer/${DATA}/"
echo "======================================================"
