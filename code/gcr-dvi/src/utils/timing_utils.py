import json
import os
import statistics
import time
from typing import Any, Callable


def synchronize_cuda() -> None:
    """Synchronize CUDA work when torch is available.

    This keeps GPU generation timing honest; without synchronization, CUDA calls
    can return before kernels finish.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed_call(fn: Callable, *args, sync_cuda: bool = False, **kwargs):
    if sync_cuda:
        synchronize_cuda()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if sync_cuda:
        synchronize_cuda()
    return result, time.perf_counter() - start


def perf_counter() -> float:
    return time.perf_counter()


def elapsed_since(start: float, sync_cuda: bool = False) -> float:
    if sync_cuda:
        synchronize_cuda()
    return time.perf_counter() - start


def _flatten_numeric(prefix: str, value: Any, out: dict[str, list[float]]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.setdefault(prefix, []).append(float(value))
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"kg_decode_per_constraint"}:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(child_prefix, child, out)


def write_timing_summary(pred_file: str, out_file: str | None = None) -> dict:
    """Summarize numeric timing fields from a predictions jsonl file."""
    values: dict[str, list[float]] = {}
    n_records = 0
    n_with_timing = 0

    if not os.path.exists(pred_file):
        return {}

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            timing = record.get("timing")
            if isinstance(timing, dict):
                n_with_timing += 1
                _flatten_numeric("", timing, values)
            inference_time = record.get("inference_time")
            if isinstance(inference_time, (int, float)) and not isinstance(inference_time, bool):
                values.setdefault("inference_time", []).append(float(inference_time))

    metrics = {}
    for key, vals in sorted(values.items()):
        if not vals:
            continue
        vals_sorted = sorted(vals)
        p90_index = max(0, int(0.9 * (len(vals_sorted) - 1)))
        metrics[key] = {
            "count": len(vals),
            "mean": sum(vals) / len(vals),
            "median": statistics.median(vals),
            "p90": vals_sorted[p90_index],
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
            "sum": sum(vals),
        }

    summary = {
        "prediction_file": pred_file,
        "unit": "seconds",
        "n_records": n_records,
        "n_with_timing": n_with_timing,
        "metrics": metrics,
    }

    if out_file is None:
        out_file = pred_file.replace("predictions.jsonl", "timing_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
