#!/bin/bash
# ============================================================
# Failure Analysis: Part 2 (5%)
# 分析 Baseline 和 DVI 的失败案例, 按 compositionality_type 分桶
# ============================================================

set -e

DATA_PATH=rmanluo
DATA=RoG-cwq
SPLIT=test
K=10
HOP=2
KG_MODEL=GCR-Meta-Llama-3.1-8B-Instruct
GENERAL_MODEL=gpt-4o-mini

BASELINE_PRED="results/GenPaths/${DATA}/${KG_MODEL}/${SPLIT}/zero-shot-group-beam-k${K}-index_len${HOP}/predictions.jsonl"
DVI_PRED="results/DVI/${DATA}/DVI-${KG_MODEL}-x-${GENERAL_MODEL}/${SPLIT}/k${K}-hop${HOP}/predictions.jsonl"

# ── 1. 分析 Baseline 失败案例 ─────────────────────────────────
echo ""
echo "======================================================"
echo "[Analysis] Baseline failure analysis"
echo "======================================================"
python workflow/failure_analysis.py \
    --pred_file ${BASELINE_PRED} \
    --data_path ${DATA_PATH} \
    --d ${DATA} \
    --split ${SPLIT} \
    --output_dir results/failure_analysis/baseline \
    --n_cases 10

# ── 2. 分析 DVI 失败案例, 并与 Baseline 对比 ─────────────────
if [ -f "${DVI_PRED}" ]; then
    echo ""
    echo "======================================================"
    echo "[Analysis] DVI failure analysis + delta vs baseline"
    echo "======================================================"
    python workflow/failure_analysis.py \
        --pred_file ${DVI_PRED} \
        --baseline_pred_file ${BASELINE_PRED} \
        --data_path ${DATA_PATH} \
        --d ${DATA} \
        --split ${SPLIT} \
        --output_dir results/failure_analysis/dvi_vs_baseline \
        --n_cases 10
else
    echo "[Analysis] DVI predictions not found at ${DVI_PRED}, skipping DVI analysis."
fi

echo ""
echo "======================================================"
echo "Analysis complete. Results:"
echo "  Baseline: results/failure_analysis/baseline/"
echo "  DVI:      results/failure_analysis/dvi_vs_baseline/"
echo "======================================================"
