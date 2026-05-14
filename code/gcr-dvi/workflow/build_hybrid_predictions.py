"""Build post-hoc hybrid predictions from baseline, DVI, and PathScorer outputs.

The router intentionally uses only prediction-time metadata, never labels. This
lets us evaluate confidence-gated variants without rerunning expensive inference.

Typical accuracy-oriented policy:
  baseline unless DVI has a small non-empty candidate set.

Typical efficiency-oriented policy:
  PathScorer for small non-empty candidate sets, otherwise baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


def load_jsonl_by_id(path: str) -> OrderedDict[str, dict[str, Any]]:
    rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def first_number(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def method_features(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "available": False,
            "num_final_candidates": None,
            "strict_intersection_size": None,
            "relaxed": None,
            "used_fallback": None,
            "mode": None,
        }

    timing = row.get("timing") or {}
    meta = row.get("dvi_meta") or {}
    intersect = meta.get("intersect_stats") or {}
    final_candidates = meta.get("final_candidates")

    num_final_candidates = first_number(
        timing.get("num_final_candidates"),
        intersect.get("final_size"),
        len(final_candidates) if isinstance(final_candidates, list) else None,
    )
    strict_intersection_size = first_number(intersect.get("strict_intersection_size"))
    n_relation_constraints = meta.get("n_relation_constraints")
    bucket = (
        f"final={num_final_candidates}|strict={strict_intersection_size}|"
        f"relaxed={intersect.get('relaxed')}|fallback={meta.get('used_fallback')}|"
        f"rel={n_relation_constraints}"
    )

    return {
        "available": True,
        "num_final_candidates": num_final_candidates,
        "strict_intersection_size": strict_intersection_size,
        "relaxed": intersect.get("relaxed"),
        "used_fallback": meta.get("used_fallback"),
        "mode": meta.get("mode"),
        "selection_strategy": meta.get("selection_strategy"),
        "n_relation_constraints": n_relation_constraints,
        "n_filter_constraints": meta.get("n_filter_constraints"),
        "metadata_bucket": bucket,
    }


def is_confident(
    features: dict[str, Any],
    *,
    max_candidates: int,
    min_candidates: int,
    require_not_relaxed: bool,
    require_not_fallback: bool,
    use_strict_intersection: bool,
) -> bool:
    if not features.get("available"):
        return False

    count_key = "strict_intersection_size" if use_strict_intersection else "num_final_candidates"
    count = features.get(count_key)
    if count is None:
        return False
    if count < min_candidates or count > max_candidates:
        return False
    if require_not_relaxed and features.get("relaxed") is True:
        return False
    if require_not_fallback and features.get("used_fallback") is True:
        return False
    return True


def choose_source(
    baseline: dict[str, Any],
    dvi: dict[str, Any] | None,
    scorer: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dvi_features = method_features(dvi)
    scorer_features = method_features(scorer)

    if args.dvi_allowed_buckets:
        dvi_confident = (
            dvi_features.get("available")
            and dvi_features.get("metadata_bucket") in args.dvi_allowed_buckets
        )
    else:
        dvi_confident = is_confident(
            dvi_features,
            max_candidates=args.dvi_max_candidates,
            min_candidates=args.min_candidates,
            require_not_relaxed=args.require_not_relaxed,
            require_not_fallback=args.require_not_fallback,
            use_strict_intersection=args.use_strict_intersection,
        )
    scorer_confident = is_confident(
        scorer_features,
        max_candidates=args.scorer_max_candidates,
        min_candidates=args.min_candidates,
        require_not_relaxed=args.require_not_relaxed,
        require_not_fallback=args.require_not_fallback,
        use_strict_intersection=args.use_strict_intersection,
    )

    source = "baseline"
    chosen = baseline

    if args.policy == "dvi_confident":
        if dvi_confident and dvi is not None:
            source = "dvi"
            chosen = dvi
    elif args.policy == "scorer_confident":
        if scorer_confident and scorer is not None:
            source = "path_scorer"
            chosen = scorer
    elif args.policy == "scorer_then_dvi":
        if scorer_confident and scorer is not None:
            source = "path_scorer"
            chosen = scorer
        elif dvi_confident and dvi is not None:
            source = "dvi"
            chosen = dvi
    elif args.policy == "dvi_then_scorer":
        if dvi_confident and dvi is not None:
            source = "dvi"
            chosen = dvi
        elif scorer_confident and scorer is not None:
            source = "path_scorer"
            chosen = scorer
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    decision = {
        "policy": args.policy,
        "source": source,
        "dvi_confident": dvi_confident,
        "path_scorer_confident": scorer_confident,
        "dvi_features": dvi_features,
        "path_scorer_features": scorer_features,
    }
    return source, chosen, decision


def load_allowed_buckets(paths: list[str], inline_buckets: list[str]) -> set[str]:
    buckets = {bucket.strip() for bucket in inline_buckets if bucket.strip()}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    buckets.add(line)
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_pred_file", required=True)
    parser.add_argument("--dvi_pred_file", default="")
    parser.add_argument("--scorer_pred_file", default="")
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--policy",
        choices=["dvi_confident", "scorer_confident", "scorer_then_dvi", "dvi_then_scorer"],
        default="dvi_confident",
    )
    parser.add_argument("--dvi_max_candidates", type=int, default=2)
    parser.add_argument("--scorer_max_candidates", type=int, default=1)
    parser.add_argument("--min_candidates", type=int, default=1)
    parser.add_argument("--require_not_relaxed", action="store_true")
    parser.add_argument("--require_not_fallback", action="store_true")
    parser.add_argument("--use_strict_intersection", action="store_true")
    parser.add_argument(
        "--dvi_allowed_bucket",
        action="append",
        default=[],
        help="DVI metadata bucket to route to DVI. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dvi_allowed_bucket_file",
        action="append",
        default=[],
        help="Text file with one DVI metadata bucket per line. # comments are ignored.",
    )
    parser.add_argument("--require_dvi", action="store_true")
    parser.add_argument("--require_scorer", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    args.dvi_allowed_buckets = load_allowed_buckets(
        args.dvi_allowed_bucket_file, args.dvi_allowed_bucket
    )

    baseline_rows = load_jsonl_by_id(args.baseline_pred_file)
    dvi_rows = load_jsonl_by_id(args.dvi_pred_file) if args.dvi_pred_file else {}
    scorer_rows = load_jsonl_by_id(args.scorer_pred_file) if args.scorer_pred_file else {}

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    skipped = Counter()
    with output_path.open("w") as fout:
        for qid, baseline in baseline_rows.items():
            if args.require_dvi and qid not in dvi_rows:
                skipped["missing_dvi"] += 1
                continue
            if args.require_scorer and qid not in scorer_rows:
                skipped["missing_scorer"] += 1
                continue
            if args.limit and sum(counts.values()) >= args.limit:
                break
            source, chosen, decision = choose_source(
                baseline=baseline,
                dvi=dvi_rows.get(qid),
                scorer=scorer_rows.get(qid),
                args=args,
            )
            counts[source] += 1
            out = dict(chosen)
            out["hybrid_meta"] = {
                **decision,
                "baseline_pred_file": args.baseline_pred_file,
                "dvi_pred_file": args.dvi_pred_file,
                "scorer_pred_file": args.scorer_pred_file,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "output_file": str(output_path),
        "n_records": sum(counts.values()),
        "source_counts": dict(counts),
        "skipped_counts": dict(skipped),
        "policy": args.policy,
        "dvi_max_candidates": args.dvi_max_candidates,
        "scorer_max_candidates": args.scorer_max_candidates,
        "min_candidates": args.min_candidates,
        "require_not_relaxed": args.require_not_relaxed,
        "require_not_fallback": args.require_not_fallback,
        "use_strict_intersection": args.use_strict_intersection,
        "dvi_allowed_buckets": sorted(args.dvi_allowed_buckets),
    }
    summary_path = output_path.with_name("hybrid_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.eval:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.utils.qa_utils import eval_result

        eval_result(str(output_path))


if __name__ == "__main__":
    main()
