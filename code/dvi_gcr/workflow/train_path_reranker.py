"""Fine-tune a small cross-encoder as a question-aware KG path reranker."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator


def load_examples(path: str) -> list[InputExample]:
    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                InputExample(
                    texts=[row["question"], row["path"]],
                    label=float(row["label"]),
                )
            )
    return examples


def load_grouped(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_grouped(model: CrossEncoder, grouped_file: str, output_file: str = "") -> dict:
    groups = load_grouped(grouped_file)
    ks = [1, 3, 5, 10]
    hit_at = {k: 0 for k in ks}
    mrr = 0.0
    evaluated = 0
    rows = []

    for group in groups:
        candidates = group.get("paths", [])
        if not candidates or not any(p.get("label") == 1 for p in candidates):
            continue
        pairs = [(group["question"], p["path"]) for p in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        labels = [int(item[0].get("label", 0)) for item in ranked]
        evaluated += 1
        first_positive_rank = next((idx + 1 for idx, label in enumerate(labels) if label == 1), None)
        if first_positive_rank:
            mrr += 1.0 / first_positive_rank
        for k in ks:
            hit_at[k] += int(any(labels[:k]))
        rows.append(
            {
                "id": group["id"],
                "first_positive_rank": first_positive_rank,
                "top_paths": [
                    {
                        "path": item[0]["path"],
                        "label": item[0]["label"],
                        "score": float(item[1]),
                    }
                    for item in ranked[:10]
                ],
            }
        )

    result = {
        "groups": evaluated,
        "mrr": mrr / evaluated if evaluated else 0.0,
        **{f"hit@{k}": hit_at[k] / evaluated if evaluated else 0.0 for k in ks},
    }
    if output_file:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"metrics": result, "examples": rows}, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--grouped_dev_file", default="")
    parser.add_argument("--model_name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--evaluation_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--eval_before_training", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_examples = load_examples(args.train_file)
    dev_examples = load_examples(args.dev_file)
    if not train_examples:
        raise ValueError(f"No training examples loaded from {args.train_file}")
    if not dev_examples:
        raise ValueError(f"No dev examples loaded from {args.dev_file}")

    model = CrossEncoder(
        args.model_name,
        num_labels=1,
        max_length=args.max_length,
        device=args.device,
    )
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        dev_examples,
        name="dev",
    )
    warmup_steps = math.ceil(len(train_loader) * args.epochs * args.warmup_ratio)

    if args.eval_before_training and args.grouped_dev_file:
        metrics = evaluate_grouped(
            model,
            args.grouped_dev_file,
            str(Path(args.output_dir) / "grouped_dev_rerank_eval_before.json"),
        )
        print("[before_training]")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        optimizer_params={"lr": args.learning_rate},
        evaluation_steps=args.evaluation_steps,
        save_best_model=True,
        use_amp=args.use_amp,
    )

    if args.grouped_dev_file:
        metrics = evaluate_grouped(
            model,
            args.grouped_dev_file,
            str(Path(args.output_dir) / "grouped_dev_rerank_eval.json"),
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
