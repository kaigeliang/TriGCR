"""Analyze complementarity between GCR baseline and DVI predictions.

This script is intentionally post-hoc: it does not create a deployable model.
It estimates how much headroom exists if a router could choose the better
prediction source per question.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(<pad>)\b", " ", text)
    return " ".join(text.split())


def match(text: str, answer: str) -> bool:
    return normalize(answer) in normalize(text)


def as_prediction_list(prediction: Any) -> list[str]:
    if isinstance(prediction, str):
        items = prediction.split("\n")
    elif isinstance(prediction, list):
        items = [str(x) for x in prediction]
    else:
        items = [str(prediction)]
    seen: dict[str, int] = {}
    for item in items:
        item = item.strip()
        if item:
            seen[item] = seen.get(item, 0) + 1
    return [item for item, _ in sorted(seen.items(), key=lambda x: x[1], reverse=True)]


def eval_acc(prediction: str, answers: list[str]) -> float:
    return sum(1.0 for a in answers if match(prediction, a)) / len(answers)


def eval_hit(prediction: str, answers: list[str]) -> float:
    return float(any(match(prediction, a) for a in answers))


def eval_hit_at1(prediction: list[str], answers: list[str]) -> float:
    first = next((p.strip() for p in prediction if p.strip()), "")
    return eval_hit(first, answers) if first else 0.0


def eval_f1(prediction: list[str], answers: list[str]) -> tuple[float, float, float]:
    if not prediction or not answers:
        return 0.0, 0.0, 0.0
    prediction_str = " ".join(prediction)
    recall = sum(1.0 for a in answers if match(prediction_str, a)) / len(answers)
    correct = 0.0
    for p in prediction:
        if any(match(p, a) for a in answers):
            correct += 1.0
    precision = correct / len(prediction)
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def score(row: dict[str, Any]) -> dict[str, float]:
    prediction = as_prediction_list(row.get("prediction", []))
    answer = list(set(row.get("ground_truth", [])))
    f1, precision, recall = eval_f1(prediction, answer)
    prediction_str = " ".join(prediction)
    return {
        "acc": eval_acc(prediction_str, answer),
        "hit": eval_hit(prediction_str, answer),
        "hit1": eval_hit_at1(prediction, answer),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def pct(x: float) -> float:
    return round(100.0 * x, 4)


def aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    n = len(scores)
    return {
        "n": n,
        "accuracy": pct(sum(s["acc"] for s in scores) / n),
        "hit": pct(sum(s["hit"] for s in scores) / n),
        "hit@1": pct(sum(s["hit1"] for s in scores) / n),
        "f1": pct(sum(s["f1"] for s in scores) / n),
        "precision": pct(sum(s["precision"] for s in scores) / n),
        "recall": pct(sum(s["recall"] for s in scores) / n),
    }


def cv_metadata_router(
    records: list[tuple[str, dict[str, float], dict[str, float], dict[str, Any]]],
    folds: int,
    min_count: int,
    margin: float,
) -> tuple[dict[str, float], int]:
    routed_scores: list[dict[str, float]] = []
    dvi_used = 0
    for fold in range(folds):
        train = [r for i, r in enumerate(records) if i % folds != fold]
        test = [r for i, r in enumerate(records) if i % folds == fold]
        stats: dict[str, list[float]] = defaultdict(list)
        for _, b, d, row in train:
            stats[bucket_key(row)].append(d["f1"] - b["f1"])
        use_dvi = {
            key
            for key, deltas in stats.items()
            if len(deltas) >= min_count and sum(deltas) / len(deltas) > margin
        }
        for _, b, d, row in test:
            if bucket_key(row) in use_dvi:
                routed_scores.append(d)
                dvi_used += 1
            else:
                routed_scores.append(b)
    return aggregate(routed_scores), dvi_used


def bucket_key(row: dict[str, Any]) -> str:
    meta = row.get("dvi_meta") or {}
    stats = meta.get("intersect_stats") or {}
    final_size = stats.get("final_size", len(meta.get("final_candidates") or []))
    strict_size = stats.get("strict_intersection_size")
    relaxed = stats.get("relaxed")
    fallback = meta.get("used_fallback")
    rel_constraints = meta.get("n_relation_constraints")
    return (
        f"final={final_size}|strict={strict_size}|relaxed={relaxed}|"
        f"fallback={fallback}|rel={rel_constraints}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--dvi", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out_json", required=True, type=Path)
    parser.add_argument("--out_md", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    dvi = load_jsonl(args.dvi)
    ids = sorted(set(baseline) & set(dvi))

    base_scores: list[dict[str, float]] = []
    dvi_scores: list[dict[str, float]] = []
    oracle_f1_scores: list[dict[str, float]] = []
    oracle_hit1_scores: list[dict[str, float]] = []
    complement = Counter()
    buckets: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = defaultdict(list)

    for qid in ids:
        b = score(baseline[qid])
        d = score(dvi[qid])
        base_scores.append(b)
        dvi_scores.append(d)
        oracle_f1_scores.append(d if d["f1"] > b["f1"] else b)
        oracle_hit1_scores.append(d if d["hit1"] > b["hit1"] else b)

        if b["hit1"] and d["hit1"]:
            complement["both_hit1"] += 1
        elif b["hit1"] and not d["hit1"]:
            complement["baseline_only_hit1"] += 1
        elif d["hit1"] and not b["hit1"]:
            complement["dvi_only_hit1"] += 1
        else:
            complement["neither_hit1"] += 1

        if d["f1"] > b["f1"]:
            complement["dvi_higher_f1"] += 1
        elif b["f1"] > d["f1"]:
            complement["baseline_higher_f1"] += 1
        else:
            complement["equal_f1"] += 1
        buckets[bucket_key(dvi[qid])].append((b, d))

    bucket_rows = []
    records: list[tuple[str, dict[str, float], dict[str, float], dict[str, Any]]] = []
    for key, pairs in buckets.items():
        if len(pairs) < 20:
            continue
        b_list = [p[0] for p in pairs]
        d_list = [p[1] for p in pairs]
        bucket_rows.append(
            {
                "bucket": key,
                "n": len(pairs),
                "baseline_hit@1": aggregate(b_list)["hit@1"],
                "dvi_hit@1": aggregate(d_list)["hit@1"],
                "baseline_f1": aggregate(b_list)["f1"],
                "dvi_f1": aggregate(d_list)["f1"],
                "dvi_minus_baseline_f1": round(
                    aggregate(d_list)["f1"] - aggregate(b_list)["f1"], 4
                ),
            }
        )
    bucket_rows.sort(key=lambda x: (-x["n"], x["bucket"]))

    for qid in ids:
        records.append((qid, score(baseline[qid]), score(dvi[qid]), dvi[qid]))
    cv_rows = []
    for min_count in (10, 20, 50, 100):
        for margin in (0.0, 0.005, 0.01, 0.02):
            metrics, dvi_used = cv_metadata_router(records, 5, min_count, margin)
            cv_rows.append(
                {
                    "folds": 5,
                    "min_count": min_count,
                    "margin": margin,
                    "dvi_used": dvi_used,
                    **metrics,
                }
            )
    cv_rows.sort(key=lambda x: (x["f1"], x["hit@1"], x["hit"]), reverse=True)

    result = {
        "label": args.label,
        "baseline_file": str(args.baseline),
        "dvi_file": str(args.dvi),
        "n_common": len(ids),
        "baseline": aggregate(base_scores),
        "dvi": aggregate(dvi_scores),
        "oracle_by_f1": aggregate(oracle_f1_scores),
        "oracle_by_hit@1": aggregate(oracle_hit1_scores),
        "complementarity_counts": dict(complement),
        "cv_metadata_router_grid": cv_rows,
        "best_cv_metadata_router": cv_rows[0] if cv_rows else None,
        "metadata_buckets_min20": bucket_rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Routing Oracle Analysis: {args.label}",
        "",
        f"Common examples: {len(ids)}",
        "",
        "| Source | Acc./Rec. | Hit | Hit@1 | F1 | Precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["baseline", "dvi", "oracle_by_f1", "oracle_by_hit@1"]:
        m = result[name]
        lines.append(
            f"| {name} | {m['accuracy']:.2f} | {m['hit']:.2f} | "
            f"{m['hit@1']:.2f} | {m['f1']:.2f} | {m['precision']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Complementarity",
            "",
            "| Pattern | Count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(complement.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## 5-Fold Metadata Router (exploratory)",
            "",
            "| Min bucket count | Margin | DVI used | Acc./Rec. | Hit | Hit@1 | F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cv_rows[:8]:
        lines.append(
            f"| {row['min_count']} | {row['margin']:.3f} | {row['dvi_used']} | "
            f"{row['accuracy']:.2f} | {row['hit']:.2f} | {row['hit@1']:.2f} | {row['f1']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## DVI Metadata Buckets (n >= 20)",
            "",
            "| Bucket | N | Baseline Hit@1 | DVI Hit@1 | Baseline F1 | DVI F1 | Delta F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bucket_rows[:30]:
        lines.append(
            f"| `{row['bucket']}` | {row['n']} | {row['baseline_hit@1']:.2f} | "
            f"{row['dvi_hit@1']:.2f} | {row['baseline_f1']:.2f} | "
            f"{row['dvi_f1']:.2f} | {row['dvi_minus_baseline_f1']:.2f} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
