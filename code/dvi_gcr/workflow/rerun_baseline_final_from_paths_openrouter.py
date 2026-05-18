"""Rerun GCR baseline Step 2 from saved path predictions via OpenRouter.

This avoids reloading the HuggingFace dataset. The saved Step-1 path files
already contain question, ground-truth answers, predicted paths, path timing,
and ground-truth paths, which are sufficient for baseline final-answer reruns.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import string
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


SAQ_RULE_INSTRUCTION = (
    "Based on the reasoning paths, please answer the given question. Please "
    "keep the answer as simple as possible and only return answers. Please "
    "return each answer in a new line."
)
MCQ_RULE_INSTRUCTION = (
    "Based on the reasoning paths, please answer the given question. Please "
    "select the answers from the given choices and return the answers only. "
    "Please return each answer in a new line."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def processed_ids(path: Path, force: bool) -> set[str]:
    if force or not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(<pad>)\b", " ", text)
    return " ".join(text.split())


def match(text: str, answer: str) -> bool:
    return normalize(answer) in normalize(text)


def prediction_list(prediction: Any) -> list[str]:
    if isinstance(prediction, str):
        items = prediction.split("\n")
    elif isinstance(prediction, list):
        items = [str(x) for x in prediction]
    else:
        items = [str(prediction)]
    return [item.strip() for item in items if item.strip()]


def score_row(row: dict[str, Any]) -> dict[str, float]:
    pred = prediction_list(row.get("prediction", []))
    answers = list(set(row.get("ground_truth", [])))
    pred_text = " ".join(pred)
    recall = (
        sum(1.0 for answer in answers if match(pred_text, answer)) / len(answers)
        if answers
        else 0.0
    )
    hit = float(any(match(pred_text, answer) for answer in answers)) if answers else 0.0
    hit1 = (
        float(any(match(pred[0], answer) for answer in answers))
        if pred and answers
        else 0.0
    )
    precision = (
        sum(1.0 for item in pred if any(match(item, answer) for answer in answers))
        / len(pred)
        if pred
        else 0.0
    )
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "acc": recall,
        "hit": hit,
        "hit@1": hit1,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def write_eval(pred_file: Path) -> None:
    rows = load_jsonl(pred_file)
    scores = [score_row(row) for row in rows]
    detailed = pred_file.with_name("detailed_eval_result.jsonl")
    with detailed.open("w", encoding="utf-8") as f:
        for row, score in zip(rows, scores):
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "prediction": prediction_list(row.get("prediction", [])),
                        "ground_truth": list(set(row.get("ground_truth", []))),
                        **score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    n = len(scores)
    metric = {key: 100.0 * sum(s[key] for s in scores) / n for key in scores[0]}
    times = [
        float(row["inference_time"])
        for row in rows
        if isinstance(row.get("inference_time"), (int, float))
    ]
    time_mean = sum(times) / len(times) if times else None
    time_total = sum(times) if times else None
    result = (
        f"Accuracy: {metric['acc']} Hit: {metric['hit']} "
        f"Hit@1: {metric['hit@1']} F1: {metric['f1']} "
        f"Precision: {metric['precision']} Recall: {metric['recall']} "
        f"InferenceTimeMeanSec: {time_mean if time_mean is not None else 'N/A'} "
        f"InferenceTimeTotalSec: {time_total if time_total is not None else 'N/A'} "
        f"TimingCoverage: {len(times)}/{n}"
    )
    pred_file.with_name("eval_result.txt").write_text(result, encoding="utf-8")
    print(result)


def dedup_keep_first(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def build_prompt(row: dict[str, Any], max_paths: int, remove_dup_path: bool) -> str:
    question = row["question"]
    if not question.endswith("?"):
        question += "?"
    paths = row.get("prediction", [])
    if isinstance(paths, str):
        paths = [paths]
    paths = [str(path) for path in paths if str(path).strip()]
    if remove_dup_path:
        paths = dedup_keep_first(paths)
    paths = paths[:max_paths]
    choices = row.get("choices") or []

    body = f"Reasoning Paths:\n{chr(10).join(paths)}\n\nQuestion:\n{question}"
    if choices:
        body += "\nChoices:\n" + "\n".join(map(str, choices))
        instruction = MCQ_RULE_INSTRUCTION
    else:
        instruction = SAQ_RULE_INSTRUCTION
    return f"{body}\n\n{instruction}"


class SimpleTokenizer:
    """Token length adapter for PromptBuilder prompt truncation."""

    def __call__(self, text: str) -> int:
        return len(text.split())


def get_prompt_builder_class():
    """Load the repo PromptBuilder without requiring qa_utils/sklearn.

    src.qa_prompt_builder imports `src.utils` for path/graph helpers. The
    add-path final-answer prompt path used here does not call those helpers, but
    importing `src.utils` normally imports qa_utils, which requires sklearn.
    Base Python has SOCKS support for OpenRouter but no sklearn, so we install a
    tiny stub for this one import.
    """
    stub = types.ModuleType("src.utils")
    sys.modules.setdefault("src.utils", stub)
    trie_stub = types.ModuleType("src.trie")

    class MarisaTrie:  # noqa: D401
        """Import stub; not used by PromptBuilder final-answer prompts."""

        pass

    trie_stub.MarisaTrie = MarisaTrie
    sys.modules.setdefault("src.trie", trie_stub)
    return importlib.import_module("src.qa_prompt_builder").PromptBuilder


def build_official_prompt(row: dict[str, Any], args: argparse.Namespace) -> str:
    paths = row.get("prediction", [])
    if isinstance(paths, str):
        paths = [paths]
    paths = [str(path) for path in paths if str(path).strip()]
    if args.remove_dup_path:
        paths = dedup_keep_first(paths)
    if args.max_paths:
        paths = paths[: args.max_paths]
    prompt_row = {
        "question": row["question"],
        "choices": row.get("choices") or [],
        "predicted_paths": paths,
        "ground_paths": row.get("ground_truth_paths", []),
    }
    prompt_builder = get_prompt_builder_class()
    builder = prompt_builder(
        add_path=True,
        add_rule=False,
        use_true=False,
        maximun_token=args.maximum_token,
        tokenize=SimpleTokenizer(),
        use_rog_prompt=False,
        each_line=True,
    )
    return builder.process_input(prompt_row)


def path_total(row: dict[str, Any]) -> float | None:
    timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
    for key in ("total", "path_generation_total"):
        value = timing.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    value = row.get("inference_time")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def call_openrouter(prompt: str, args: argparse.Namespace) -> tuple[str, float, str]:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or OPENROUTER_API_KEY is required")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    url = base_url.rstrip("/") + "/chat/completions"
    proxies = None
    if args.proxy:
        proxies = {"http": args.proxy, "https": args.proxy}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": args.referer,
        "X-OpenRouter-Title": args.title,
    }
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    last_error = ""
    for attempt in range(args.retry + 1):
        start = time.perf_counter()
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                proxies=proxies,
                timeout=args.timeout,
            )
            elapsed = time.perf_counter() - start
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                model = data.get("model", args.model_name)
                return content, elapsed, model
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.retry:
            time.sleep(min(args.retry_sleep * (attempt + 1), 30))
    raise RuntimeError(last_error)


def make_prediction(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if args.use_official_prompt_builder:
        prompt = build_official_prompt(row, args)
    else:
        prompt = build_prompt(row, args.max_paths, args.remove_dup_path)
    sample_start = time.perf_counter()
    try:
        prediction, api_time, resolved_model = call_openrouter(prompt, args)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {row.get('id')}: {exc}", flush=True)
        return None
    final_total = time.perf_counter() - sample_start
    kg_time = path_total(row)
    total = final_total + kg_time if kg_time is not None else final_total
    return {
        "id": row["id"],
        "question": row["question"],
        "prediction": prediction,
        "ground_truth": row.get("ground_truth", []),
        "input": prompt,
        "inference_time": total,
        "timing": {
            "schema_version": 1,
            "unit": "seconds",
            "method": "baseline_final_answer_openrouter_from_paths",
            "total": total,
            "final_answer_total": final_total,
            "final_api_time": api_time,
            "path_generation_total": kg_time,
            "path_generation": row.get("timing", {}),
            "num_api_calls": 1,
            "model_name": args.model_name,
            "resolved_model_name": resolved_model,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="gpt-4o-mini")
    parser.add_argument("--proxy", default="socks5h://127.0.0.1:1080")
    parser.add_argument("--referer", default="https://localhost")
    parser.add_argument("--title", default="dvi-gcr-part2-baseline")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_paths", type=int, default=10)
    parser.add_argument("--maximum_token", type=int, default=128000)
    parser.add_argument("--use_official_prompt_builder", action="store_true")
    parser.add_argument(
        "--remove_dup_path",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Match baseline Step 2 default by removing duplicate paths.",
    )
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.reasoning_path)
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = args.output_dir / "predictions.jsonl"
    done = processed_ids(pred_file, args.force)
    mode = "w" if args.force or not pred_file.exists() else "a"
    pending = [row for row in rows if row["id"] not in done]

    args_path = args.output_dir / "args.json"
    args_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"input_rows={len(rows)} pending={len(pending)} output={pred_file}", flush=True)
    with pred_file.open(mode, encoding="utf-8") as f:
        if args.n <= 1:
            iterator = (make_prediction(row, args) for row in pending)
        else:
            pool = ThreadPoolExecutor(max_workers=args.n)
            iterator = pool.map(lambda row: make_prediction(row, args), pending)
        for i, result in enumerate(iterator, 1):
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            if i % 25 == 0:
                print(f"processed={i}/{len(pending)}", flush=True)
        if args.n > 1:
            pool.shutdown(wait=True)

    if pred_file.exists() and pred_file.stat().st_size > 0:
        write_eval(pred_file)


if __name__ == "__main__":
    main()
