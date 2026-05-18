"""Build question-path training data for a lightweight KGQA verifier.

The verifier target is deliberately simple: given a question and one DFS
candidate path, predict whether the path ends at a gold answer entity. This
matches the entity/relation-level decoding direction while avoiding labels from
noisy final-answer generations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

import src.utils as utils


def normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def path_endpoint(path: list[tuple[str, str, str]]) -> str:
    return path[-1][2] if path else ""


def path_relations(path: list[tuple[str, str, str]]) -> tuple[str, ...]:
    return tuple(edge[1] for edge in path)


def is_positive_endpoint(endpoint: str, answers: Iterable[str], answer_entities: Iterable[str]) -> bool:
    endpoint_norm = normalize(endpoint)
    answer_entity_norms = {normalize(a) for a in answer_entities}
    answer_norms = {normalize(a) for a in answers}
    return endpoint_norm in answer_entity_norms or endpoint_norm in answer_norms


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * pct)))
    return float(values[idx])


def rank_negatives(
    negatives: list[dict],
    positive_paths: list[dict],
    strategy: str,
    rng: random.Random,
) -> list[dict]:
    if strategy == "all":
        return negatives
    if strategy == "random":
        shuffled = negatives[:]
        rng.shuffle(shuffled)
        return shuffled

    positive_lens = {item["path_len"] for item in positive_paths}
    positive_rels = set()
    positive_relseqs = set()
    for item in positive_paths:
        rels = tuple(item["relation_sequence"])
        positive_relseqs.add(rels)
        positive_rels.update(rels)

    def hard_score(item: dict) -> tuple[int, int, int, str]:
        rels = tuple(item["relation_sequence"])
        same_len = int(item["path_len"] in positive_lens)
        shared_rel = len(set(rels) & positive_rels)
        same_relseq = int(rels in positive_relseqs)
        return (same_relseq, same_len, shared_rel, item["path"])

    return sorted(negatives, key=hard_score, reverse=True)


def build_records_for_sample(
    sample: dict,
    index_path_length: int,
    undirected: bool,
    max_negatives: int,
    negative_strategy: str,
    rng: random.Random,
    include_no_positive: bool,
) -> tuple[list[dict], dict | None]:
    graph = utils.build_graph(sample["graph"], undirected)
    candidate_paths = utils.dfs(graph, sample["q_entity"], index_path_length)
    candidate_paths = sorted(candidate_paths, key=utils.path_to_string)

    records = []
    for path in candidate_paths:
        path_str = utils.path_to_string(path)
        endpoint = path_endpoint(path)
        rels = path_relations(path)
        label = int(is_positive_endpoint(endpoint, sample["answer"], sample["a_entity"]))
        records.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "q_entity": sample["q_entity"],
                "answer": sample["answer"],
                "a_entity": sample["a_entity"],
                "path": path_str,
                "endpoint": endpoint,
                "relation_sequence": list(rels),
                "path_len": len(path),
                "label": label,
            }
        )

    positives = [r for r in records if r["label"] == 1]
    negatives = [r for r in records if r["label"] == 0]
    if not positives and not include_no_positive:
        return [], None

    negatives = rank_negatives(negatives, positives, negative_strategy, rng)
    if max_negatives >= 0:
        negatives = negatives[:max_negatives]

    selected = positives + negatives
    for rank, record in enumerate(selected):
        record["candidate_count"] = len(records)
        record["positive_count_for_question"] = len(positives)
        record["selected_negative_count_for_question"] = len(negatives)
        record["selected_rank"] = rank

    group = {
        "id": sample["id"],
        "question": sample["question"],
        "q_entity": sample["q_entity"],
        "answer": sample["answer"],
        "a_entity": sample["a_entity"],
        "candidate_count": len(records),
        "positive_count": len(positives),
        "selected_negative_count": len(negatives),
        "paths": [
            {
                "path": r["path"],
                "endpoint": r["endpoint"],
                "relation_sequence": r["relation_sequence"],
                "path_len": r["path_len"],
                "label": r["label"],
            }
            for r in selected
        ],
    }
    return selected, group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="rmanluo")
    parser.add_argument("--d", default="RoG-cwq")
    parser.add_argument("--split", default="train[:1000]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--grouped_output", default="")
    parser.add_argument("--index_path_length", type=int, default=2)
    parser.add_argument("--undirected", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max_negatives_per_question", type=int, default=50)
    parser.add_argument(
        "--negative_strategy",
        choices=["hard", "random", "all"],
        default="hard",
    )
    parser.add_argument("--include_no_positive", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = load_dataset(os.path.join(args.data_path, args.d), split=args.split)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_path = Path(args.grouped_output) if args.grouped_output else None
    if grouped_path:
        grouped_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    candidate_counts = []
    positive_counts = []
    selected_counts = []

    with output_path.open("w") as fout:
        grouped_fout = grouped_path.open("w") if grouped_path else None
        try:
            for sample in dataset:
                stats["questions"] += 1
                records, group = build_records_for_sample(
                    sample=sample,
                    index_path_length=args.index_path_length,
                    undirected=args.undirected,
                    max_negatives=args.max_negatives_per_question,
                    negative_strategy=args.negative_strategy,
                    rng=rng,
                    include_no_positive=args.include_no_positive,
                )
                if group is None:
                    stats["skipped_no_positive"] += 1
                    continue

                stats["questions_with_positive"] += int(group["positive_count"] > 0)
                stats["examples"] += len(records)
                stats["positives"] += sum(r["label"] for r in records)
                stats["negatives"] += sum(1 - r["label"] for r in records)
                candidate_counts.append(group["candidate_count"])
                positive_counts.append(group["positive_count"])
                selected_counts.append(len(records))

                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                if grouped_fout:
                    grouped_fout.write(json.dumps(group, ensure_ascii=False) + "\n")
        finally:
            if grouped_fout:
                grouped_fout.close()

    summary = {
        "data_path": args.data_path,
        "dataset": args.d,
        "split": args.split,
        "output": str(output_path),
        "grouped_output": str(grouped_path) if grouped_path else "",
        "index_path_length": args.index_path_length,
        "undirected": args.undirected,
        "max_negatives_per_question": args.max_negatives_per_question,
        "negative_strategy": args.negative_strategy,
        "questions": stats["questions"],
        "questions_with_positive": stats["questions_with_positive"],
        "skipped_no_positive": stats["skipped_no_positive"],
        "examples": stats["examples"],
        "positives": stats["positives"],
        "negatives": stats["negatives"],
        "candidate_count_avg": (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
        "candidate_count_p50": percentile(candidate_counts, 0.50),
        "candidate_count_p95": percentile(candidate_counts, 0.95),
        "positive_count_avg": (sum(positive_counts) / len(positive_counts)) if positive_counts else 0.0,
        "selected_count_avg": (sum(selected_counts) / len(selected_counts)) if selected_counts else 0.0,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
