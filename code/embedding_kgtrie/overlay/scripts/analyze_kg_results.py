#!/usr/bin/env python
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean

from datasets import load_dataset
from src.utils.qa_utils import eval_f1, eval_hit, extract_topk_prediction


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_question_type(question, row=None):
    if row and row.get("compositionality_type"):
        return row["compositionality_type"]
    text = question.lower()
    if re.search(r"\b(before|after|during|when|year|date|first|last|previous|next)\b", text):
        return "temporal"
    if re.search(r"\b(highest|largest|smallest|longest|shortest|most|least|top|first|last|number of|how many)\b", text):
        return "comparative_or_count"
    if re.search(r"\b(and|both|also|as well as|while|who .* and |that .* and )\b", text):
        return "conjunctive_multi_constraint"
    if len(row.get("q_entity", [])) > 1 if row else False:
        return "multi_entity"
    if re.search(r"\b(type|kind|category|profession|occupation|nationality)\b", text):
        return "type_constraint"
    return "other"


def parse_prediction_answers(prediction, topk):
    prediction = extract_topk_prediction(prediction, topk)
    answers = []
    paths = []
    for item in prediction:
        answer = item.split("# Answer:\n")[-1].strip()
        path = item.split("# Answer:\n")[0].split("# Reasoning Path:\n")[-1].strip()
        answers.append(answer)
        paths.append(path)
    return prediction, answers, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--dataset", default="rmanluo/RoG-cwq")
    parser.add_argument("--split", default="test[:20]")
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_failures", type=int, default=10)
    args = parser.parse_args()

    rows = load_dataset(args.dataset, split=args.split)
    by_id = {row["id"]: row for row in rows}
    predictions = load_jsonl(args.prediction)

    summary = defaultdict(lambda: {"n": 0, "hit": [], "f1": [], "precision": [], "recall": []})
    failures = []
    overall = {"hit": [], "f1": [], "precision": [], "recall": []}

    for pred in predictions:
        row = by_id.get(pred["id"], {})
        question = pred.get("question") or row.get("question", "")
        qtype = infer_question_type(question, row)
        _, answers, paths = parse_prediction_answers(pred["prediction"], args.topk)
        gold = list(set(pred.get("ground_truth") or row.get("answer", [])))
        f1, precision, recall = eval_f1(answers, gold)
        hit = eval_hit(" ".join(pred["prediction"] if isinstance(pred["prediction"], list) else [pred["prediction"]]), gold)

        bucket = summary[qtype]
        bucket["n"] += 1
        for key, value in [("hit", hit), ("f1", f1), ("precision", precision), ("recall", recall)]:
            bucket[key].append(value)
            overall[key].append(value)

        if not hit:
            failures.append(
                {
                    "id": pred["id"],
                    "type": qtype,
                    "question": question,
                    "q_entity": row.get("q_entity", []),
                    "gold": gold,
                    "predicted_answers": answers,
                    "predicted_paths": paths[:3],
                }
            )

    lines = []
    lines.append("# KG Result Analysis")
    lines.append("")
    lines.append(f"Prediction file: `{args.prediction}`")
    lines.append(f"Dataset: `{args.dataset}` split `{args.split}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| n | Hit@1/Hit | F1 | Precision | Recall |")
    lines.append("|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {len(predictions)} | {mean(overall['hit'])*100:.2f} | {mean(overall['f1'])*100:.2f} | {mean(overall['precision'])*100:.2f} | {mean(overall['recall'])*100:.2f} |"
    )
    lines.append("")
    lines.append("## By Question Type")
    lines.append("")
    lines.append("| Type | n | Hit | F1 | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for qtype, stats in sorted(summary.items(), key=lambda item: (-item[1]["n"], item[0])):
        lines.append(
            f"| {qtype} | {stats['n']} | {mean(stats['hit'])*100:.2f} | {mean(stats['f1'])*100:.2f} | {mean(stats['precision'])*100:.2f} | {mean(stats['recall'])*100:.2f} |"
        )
    lines.append("")
    lines.append("## Failure Cases")
    lines.append("")
    for item in failures[: args.max_failures]:
        lines.append(f"### {item['id']} ({item['type']})")
        lines.append(f"- Question: {item['question']}")
        lines.append(f"- Topic entities: {item['q_entity']}")
        lines.append(f"- Gold: {item['gold']}")
        lines.append(f"- Predicted answers: {item['predicted_answers']}")
        lines.append("- Top predicted paths:")
        for path in item["predicted_paths"]:
            lines.append(f"  - {path}")
        lines.append("")

    content = "\n".join(lines)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
    print(content)


if __name__ == "__main__":
    main()
