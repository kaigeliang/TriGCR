"""
Failure Analysis Script — Part 2 (5%)

分析 GCR baseline 在 CWQ 上的失败案例, 按 compositionality_type 分桶统计.

输出:
  1. 控制台: 各类别的 Hit@1 / F1 / 样本数
  2. results/failure_analysis/bucket_stats.json   — 各桶指标
  3. results/failure_analysis/failure_cases.json  — 失败样本 (含推理 trace)
  4. results/failure_analysis/summary.txt         — 人类可读摘要 (直接用于报告)

Usage:
  # 分析 baseline 失败案例
  python workflow/failure_analysis.py \
    --pred_file results/GenPaths/RoG-cwq/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10-index_len2/predictions.jsonl \
    --data_path rmanluo --d RoG-cwq --split test \
    --output_dir results/failure_analysis/baseline

  # 对比 DVI vs Baseline
  python workflow/failure_analysis.py \
    --pred_file results/DVI/RoG-cwq/.../predictions.jsonl \
    --baseline_pred_file results/GenPaths/.../predictions.jsonl \
    --data_path rmanluo --d RoG-cwq --split test \
    --output_dir results/failure_analysis/dvi_vs_baseline
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional
from datasets import load_dataset
from src.utils.qa_utils import eval_f1, eval_hit, eval_hit_at1


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_predictions(pred_file: str) -> Dict[str, dict]:
    """Load prediction JSONL → {id: record}."""
    preds = {}
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            preds[r["id"]] = r
    return preds


def parse_prediction(record: dict) -> List[str]:
    """Normalize prediction field to List[str]."""
    pred = record.get("prediction", [])
    if isinstance(pred, str):
        return [p.strip() for p in pred.split("\n") if p.strip()]
    if isinstance(pred, list):
        flat = []
        for item in pred:
            if isinstance(item, str):
                flat.extend([p.strip() for p in item.split("\n") if p.strip()])
        return flat
    return []


def parse_answer(record: dict) -> List[str]:
    """Normalize answer/ground_truth to List[str]."""
    ans = record.get("answer") or record.get("ground_truth") or []
    if isinstance(ans, str):
        return [ans]
    return list(set(ans))


def score_sample(pred_list: List[str], answer_list: List[str]) -> dict:
    f1, precision, recall = eval_f1(pred_list, answer_list)
    any_hit = eval_hit(" ".join(pred_list), answer_list)
    hit1 = eval_hit_at1(pred_list, answer_list)
    return {"f1": f1, "precision": precision, "recall": recall, "hit": hit1, "any_hit": any_hit}


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze(
    pred_file: str,
    dataset,
    output_dir: str,
    baseline_preds: Optional[Dict] = None,
    n_cases_per_bucket: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    preds = load_predictions(pred_file)

    # Bucket structure: compositionality_type → list of per-sample results
    buckets: Dict[str, list] = defaultdict(list)
    failure_cases: Dict[str, list] = defaultdict(list)
    overall = []

    for sample in dataset:
        qid  = sample["id"]
        ctype = sample.get("compositionality_type", "unknown")
        answer_list = list(set(sample.get("answer", [])))

        if qid not in preds:
            continue

        record    = preds[qid]
        pred_list = parse_prediction(record)
        scores    = score_sample(pred_list, answer_list)

        entry = {
            "id":       qid,
            "ctype":    ctype,
            "question": sample["question"],
            "answer":   answer_list,
            "prediction": pred_list,
            **scores,
        }

        buckets[ctype].append(entry)
        overall.append(entry)

        # Collect failure cases (hit == 0)
        if scores["hit"] == 0 and len(failure_cases[ctype]) < n_cases_per_bucket:
            case = {
                "id":           qid,
                "question":     sample["question"],
                "q_entity":     sample.get("q_entity", []),
                "answer":       answer_list,
                "prediction":   pred_list,
                "scores":       scores,
            }
            # Include predicted paths if available
            if "predicted_paths" in record:
                case["predicted_paths"] = record["predicted_paths"][:5]
            # Include DVI metadata if available
            if "dvi_meta" in record:
                case["dvi_meta"] = record["dvi_meta"]
            failure_cases[ctype].append(case)

    # Compare with baseline if provided
    baseline_buckets: Dict[str, list] = defaultdict(list)
    if baseline_preds:
        for sample in dataset:
            qid   = sample["id"]
            ctype = sample.get("compositionality_type", "unknown")
            answer_list = list(set(sample.get("answer", [])))
            if qid not in baseline_preds:
                continue
            record    = baseline_preds[qid]
            pred_list = parse_prediction(record)
            scores    = score_sample(pred_list, answer_list)
            baseline_buckets[ctype].append(scores)

    # ── Compute bucket stats ──────────────────────────────────────────────────
    bucket_stats = {}
    all_ctypes   = sorted(set(list(buckets.keys()) + list(baseline_buckets.keys())))

    for ctype in all_ctypes:
        items = buckets.get(ctype, [])
        n     = len(items)
        if n == 0:
            continue
        avg_f1  = sum(x["f1"]  for x in items) / n
        avg_hit = sum(x["hit"] for x in items) / n
        avg_any_hit = sum(x["any_hit"] for x in items) / n
        avg_p   = sum(x["precision"] for x in items) / n
        avg_r   = sum(x["recall"]    for x in items) / n

        stat = {
            "n_total":     n,
            "n_correct":   sum(1 for x in items if x["hit"] == 1),
            "n_incorrect": sum(1 for x in items if x["hit"] == 0),
            "hit@1":       round(avg_hit, 4),
            "any_hit":      round(avg_any_hit, 4),
            "f1":          round(avg_f1,  4),
            "precision":   round(avg_p,   4),
            "recall":      round(avg_r,   4),
        }

        # Delta vs baseline
        if ctype in baseline_buckets:
            bn   = len(baseline_buckets[ctype])
            b_h  = sum(x["hit"] for x in baseline_buckets[ctype]) / bn
            b_any_h = sum(x["any_hit"] for x in baseline_buckets[ctype]) / bn
            b_f1 = sum(x["f1"]  for x in baseline_buckets[ctype]) / bn
            stat["baseline_hit@1"] = round(b_h,  4)
            stat["baseline_any_hit"] = round(b_any_h, 4)
            stat["baseline_f1"]    = round(b_f1, 4)
            stat["delta_hit@1"]    = round(avg_hit - b_h,  4)
            stat["delta_f1"]       = round(avg_f1  - b_f1, 4)

        bucket_stats[ctype] = stat

    # Overall stats
    n_all = len(overall)
    if n_all > 0:
        bucket_stats["_overall"] = {
            "n_total":   n_all,
            "hit@1":     round(sum(x["hit"] for x in overall) / n_all, 4),
            "any_hit":    round(sum(x["any_hit"] for x in overall) / n_all, 4),
            "f1":        round(sum(x["f1"]  for x in overall) / n_all, 4),
            "precision": round(sum(x["precision"] for x in overall) / n_all, 4),
            "recall":    round(sum(x["recall"]    for x in overall) / n_all, 4),
        }

    # ── Save outputs ──────────────────────────────────────────────────────────
    stats_path = os.path.join(output_dir, "bucket_stats.json")
    with open(stats_path, "w") as f:
        json.dump(bucket_stats, f, indent=2)

    cases_path = os.path.join(output_dir, "failure_cases.json")
    with open(cases_path, "w") as f:
        json.dump(failure_cases, f, indent=2, ensure_ascii=False)

    # ── Human-readable summary ────────────────────────────────────────────────
    summary_lines = [
        "=" * 60,
        "FAILURE ANALYSIS SUMMARY",
        f"Predictions: {pred_file}",
        "=" * 60,
        "",
        f"{'Type':<20} {'N':>6} {'Hit@1':>8} {'F1':>8}",
        "-" * 46,
    ]

    for ctype in sorted(k for k in bucket_stats if k != "_overall"):
        s = bucket_stats[ctype]
        line = f"{ctype:<20} {s['n_total']:>6} {s['hit@1']:>8.4f} {s['f1']:>8.4f}"
        if "delta_hit@1" in s:
            delta_h  = s["delta_hit@1"]
            delta_f1 = s["delta_f1"]
            sign_h  = "+" if delta_h  >= 0 else ""
            sign_f1 = "+" if delta_f1 >= 0 else ""
            line += f"   (Δhit {sign_h}{delta_h:.4f}, Δf1 {sign_f1}{delta_f1:.4f})"
        summary_lines.append(line)

    summary_lines.append("-" * 46)
    ov = bucket_stats.get("_overall", {})
    summary_lines.append(
        f"{'Overall':<20} {ov.get('n_total',''):>6} "
        f"{ov.get('hit@1', 0):>8.4f} {ov.get('f1', 0):>8.4f}"
    )
    summary_lines += ["", "Failure cases saved to:", f"  {cases_path}", ""]

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\n[Analysis] Stats  → {stats_path}")
    print(f"[Analysis] Cases  → {cases_path}")
    print(f"[Analysis] Summary→ {summary_path}")

    return bucket_stats, failure_cases


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print(f"[Analysis] Loading dataset: {args.data_path}/{args.d} ({args.split})")
    dataset = load_dataset(os.path.join(args.data_path, args.d), split=args.split)

    baseline_preds = None
    if args.baseline_pred_file:
        print(f"[Analysis] Loading baseline: {args.baseline_pred_file}")
        baseline_preds = load_predictions(args.baseline_pred_file)

    analyze(
        pred_file=args.pred_file,
        dataset=dataset,
        output_dir=args.output_dir,
        baseline_preds=baseline_preds,
        n_cases_per_bucket=args.n_cases,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred_file",          required=True,
                   help="Path to predictions.jsonl to analyze")
    p.add_argument("--baseline_pred_file", default=None,
                   help="(Optional) Baseline predictions.jsonl for delta comparison")
    p.add_argument("--data_path",  default="rmanluo")
    p.add_argument("--d", "-d",    default="RoG-cwq")
    p.add_argument("--split",      default="test")
    p.add_argument("--output_dir", default="results/failure_analysis")
    p.add_argument("--n_cases",    type=int, default=10,
                   help="Max failure cases to save per compositionality_type bucket")
    args = p.parse_args()
    main(args)
