"""
DVI (Decompose-Verify-Intersect) Inference Pipeline

Steps:
  1. [Decompose]  General LLM splits complex query into atomic constraints (JSON)
  2. [Verify]     Per-constraint path retrieval:
       Mode A: KG-specialized LLM decodes each constraint via mini-Trie
       Mode B: --path_scorer ranks DFS paths with bi-encoder + cross-encoder
  3. [Intersect]  Python set.intersection() produces final candidates (zero hallucination)
  4. [Answer]     General LLM generates final answer given candidates + evidence paths

Usage:
  # Mode A: original KG-LLM verifier
  python workflow/predict_dvi.py \
    --d RoG-cwq --split test \
    --kg_model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
    --general_model_name gpt-4o-mini \
    --k 10 --index_path_length 2

  # Mode B: v2 PathScorer verifier
  python workflow/predict_dvi.py \
    --d RoG-cwq --split test \
    --general_model_name gpt-4o-mini \
    --index_path_length 2 \
    --path_scorer --bi_k 100 --cross_k 10
"""

import os
import json
import copy
import argparse
import sys
import re
from types import SimpleNamespace
from tqdm import tqdm
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from src.llms import get_registed_model
from src.qa_prompt_builder import DVIPromptBuilder
from src.dvi import QueryDecomposer, CandidateIntersector, refine_candidates
from src.utils.qa_utils import eval_result
from src.utils import path_to_string
from src.utils.timing_utils import elapsed_since, perf_counter, timed_call, write_timing_summary

# ── Final answer prompts ──────────────────────────────────────────────────────

FINAL_ANSWER_WITH_CANDIDATES = """Based on the reasoning paths below, answer the original question.
The candidate entities are KG entities collected from path endpoints; they may be intermediate or noisy, so do not restrict your answer to the candidate list.
Return all possible final answers, one per line. Keep answers concise (entity names only).

# Question:
{question}

# Reasoning Paths:
{evidence_paths}

# Candidate or intermediate KG entities:
{candidates}

Answer:"""

FINAL_ANSWER_NO_CANDIDATES = """Based on the reasoning paths below, answer the question.
Return all possible answers, one per line. Keep answers concise (entity names only).

# Question:
{question}

# Reasoning Paths:
{evidence_paths}

Answer:"""


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_output_file(path: str, force: bool = False):
    if not os.path.exists(path) or force:
        return open(path, "w"), []
    with open(path, "r") as f:
        processed = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    processed.append(json.loads(line)["id"])
                except Exception:
                    pass
    return open(path, "a"), processed


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def clean_path_for_evidence(path: str) -> str:
    """Keep the KG path itself, not the KG model's path-local answer block."""
    clean = path.replace("<PATH>", "").replace("</PATH>", "").strip()
    clean = re.split(r"\n\s*#\s*Answer\s*:|\n\s*Answer\s*:", clean, maxsplit=1)[0].strip()
    return clean


def format_paths_for_display(
    paths_per_constraint: Dict[str, List[str]],
    max_per: int = 3,
    extra_paths: Optional[List[str]] = None,
) -> str:
    lines = []
    for cid, paths in paths_per_constraint.items():
        lines.append(f"[{cid}]")
        for p in dedupe_keep_order(paths)[:max_per]:
            clean = clean_path_for_evidence(p)
            lines.append(f"  {clean}")
    if extra_paths:
        lines.append("[answer-aware expansion]")
        for p in dedupe_keep_order(extra_paths)[:max_per]:
            lines.append(f"  # Reasoning Path:\n{clean_path_for_evidence(p)}")
    return "\n".join(lines) if lines else "(no paths found)"


def parse_answer_lines(text: str) -> List[str]:
    text = text.strip()
    text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE).strip()
    return [line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t")]


def make_kg_args(base_args) -> SimpleNamespace:
    """Create args namespace for the KG-specialized LLM."""
    a = copy.copy(base_args)
    a.model_name = base_args.kg_model_name
    a.model_path = base_args.kg_model_path
    # HfCausalModel needs these; ensure they exist with defaults
    if not hasattr(a, "maximun_token"):
        a.maximun_token = 4096
    if not hasattr(a, "max_new_tokens"):
        a.max_new_tokens = 512
    if not hasattr(a, "chat_model"):
        a.chat_model = True
    if not hasattr(a, "use_assistant_model"):
        a.use_assistant_model = False
    if not hasattr(a, "assistant_model_path"):
        a.assistant_model_path = None
    return a


def make_general_args(base_args) -> SimpleNamespace:
    """Create args namespace for the general LLM (ChatGPT)."""
    a = SimpleNamespace()
    a.model_name = base_args.general_model_name
    a.retry = base_args.retry
    a.model_path = "None"      # ChatGPT.add_args sets this default
    a.dtype = "bf16"
    a.quant = "none"
    return a


def should_use_global_trie(
    decomposed: dict,
    rel_constraints: List[dict],
    filter_constraints: List[dict],
    decompose_meta: dict,
) -> bool:
    """Use baseline-style global Trie for simple cases.

    DVI's intersection is most useful when there are multiple relation or
    attribute constraints. For a single relation constraint, mini-Trie decoding
    has no real intersection advantage and can lose helpful paths from other
    query entities.
    """
    if len(rel_constraints) <= 1 and len(filter_constraints) == 0:
        return True
    if decompose_meta.get("fallback_reason") == "single_entity":
        return True
    return False


def get_path_scorer_paths(
    data: dict,
    prompt_builder: DVIPromptBuilder,
    rel_constraints: List[dict],
    q_entities: List[str],
    question: str,
    use_global_paths: bool = False,
) -> tuple[Dict[str, List[str]], List[dict], dict, str]:
    """Enumerate raw paths for PathScorer while matching DVI fallback behavior."""
    if use_global_paths:
        paths = prompt_builder.get_fallback_paths(data)
        cid = "c_global"
        return (
            {cid: paths} if paths else {},
            [{
                "id": cid,
                "type": "relation",
                "anchor": ", ".join(q_entities),
                "hint": question,
            }],
            {
                cid: {
                    "anchor": ", ".join(q_entities),
                    "n_paths": len(paths),
                    "fallback": "selective_global_paths",
                }
            },
            "selective_global_paths",
        )

    raw_paths = prompt_builder.get_per_constraint_paths(data, rel_constraints)
    path_stats = getattr(prompt_builder, "last_path_stats", {})
    if raw_paths:
        return raw_paths, rel_constraints, path_stats, "path_scorer_per_constraint"

    fallback_paths = prompt_builder.get_fallback_paths(data)
    if not fallback_paths:
        return {}, rel_constraints, path_stats, "empty_paths"

    cid = "c_fallback"
    return (
        {cid: fallback_paths},
        [{
            "id": cid,
            "type": "relation",
            "anchor": ", ".join(q_entities),
            "hint": question,
        }],
        {
            cid: {
                "anchor": ", ".join(q_entities),
                "n_paths": len(fallback_paths),
                "fallback": True,
            }
        },
        "empty_paths_global_fallback",
    )


# ── Per-sample DVI inference ──────────────────────────────────────────────────

def run_dvi_on_sample(
    data: dict,
    kg_model,
    general_model,
    path_scorer,
    prompt_builder: DVIPromptBuilder,
    decomposer: QueryDecomposer,
    intersector: CandidateIntersector,
    args,
) -> Optional[dict]:

    sample_start = perf_counter()
    question = data["question"]
    if not question.endswith("?"):
        question += "?"
    answer   = data["answer"]
    qid      = data["id"]
    q_entities = data.get("q_entity", [])

    # ── Step 1: Decompose ────────────────────────────────────────────────────
    decomposed, decompose_time = timed_call(decomposer.decompose, question, q_entities)
    decompose_meta = getattr(decomposer, "last_call_metadata", {})
    rel_constraints    = decomposer.get_relation_constraints(decomposed)
    filter_constraints = decomposer.get_filter_constraints(decomposed)
    selection_strategy = "per_constraint"

    # ── Step 2: Verify — KG-LLM mini-Tries or v2 PathScorer ─────────────────
    trie_start = perf_counter()
    use_global_trie = bool(args.selective_fallback) and should_use_global_trie(
        decomposed,
        rel_constraints,
        filter_constraints,
        decompose_meta,
    )
    paths_per_constraint: Dict[str, List[str]] = {}
    kg_decode_time = 0.0
    kg_decode_per_constraint = {}
    num_kg_decode_calls = 0
    path_scorer_time = 0.0
    constraint_prompt_build_time = 0.0

    if args.path_scorer:
        raw_paths, rel_constraints, trie_stats, selection_strategy = get_path_scorer_paths(
            data,
            prompt_builder,
            rel_constraints,
            q_entities,
            question,
            use_global_paths=use_global_trie,
        )
        trie_build_time = elapsed_since(trie_start)
        if not raw_paths:
            return None
        hints_per_constraint = {
            c.get("id", c.get("anchor", "")): c.get("hint", "")
            for c in rel_constraints
        }
        paths_per_constraint, path_scorer_time = timed_call(
            path_scorer.score_constraints,
            raw_paths,
            question,
            hints_per_constraint,
            sync_cuda=path_scorer.device == "cuda",
        )
    else:
        if use_global_trie:
            fallback_trie = prompt_builder.build_fallback_trie(data)
            if fallback_trie is None:
                return None
            cid = "c_global"
            tries_per_constraint = {cid: fallback_trie}
            trie_stats = {
                cid: {
                    "anchor": ", ".join(q_entities),
                    "n_paths": None,
                    "fallback": "selective_global_trie",
                }
            }
            rel_constraints = [{
                "id": cid,
                "type": "relation",
                "anchor": ", ".join(q_entities),
                "hint": question,
            }]
            selection_strategy = "selective_global_trie"
        else:
            tries_per_constraint = prompt_builder.get_per_constraint_tries(data, rel_constraints)
            trie_stats = getattr(prompt_builder, "last_trie_stats", {})

        # If no tries built (anchor entities not in graph), fall back to global Trie
        if not tries_per_constraint:
            fallback_trie = prompt_builder.build_fallback_trie(data)
            if fallback_trie is None:
                return None
            tries_per_constraint = {"c_fallback": fallback_trie}
            trie_stats = {
                "c_fallback": {
                    "anchor": ", ".join(q_entities),
                    "n_paths": None,
                    "fallback": True,
                }
            }
            rel_constraints = [{
                "id": "c_fallback",
                "type": "relation",
                "anchor": ", ".join(q_entities),
                "hint": question,
            }]
            selection_strategy = "empty_trie_global_fallback"
        trie_build_time = elapsed_since(trie_start)

        token_start = perf_counter()
        start_tok = kg_model.tokenizer.convert_tokens_to_ids(prompt_builder.PATH_START_TOKEN)
        end_tok   = kg_model.tokenizer.convert_tokens_to_ids(prompt_builder.PATH_END_TOKEN)
        constraint_prompt_build_time = elapsed_since(token_start)

        for c in rel_constraints:
            cid    = c.get("id", c.get("anchor", "c?"))
            anchor = c.get("anchor", "")
            hint   = c.get("hint", "")
            trie   = tries_per_constraint.get(cid)
            if trie is None:
                continue

            prompt_start = perf_counter()
            prompt_text  = prompt_builder.build_constraint_prompt(question, anchor, hint, [anchor])
            model_input  = kg_model.prepare_model_prompt(prompt_text)
            constraint_prompt_build_time += elapsed_since(prompt_start)

            raw, decode_time = timed_call(
                kg_model.generate_sentence,
                model_input,
                trie,
                start_token_ids=start_tok,
                end_token_ids=end_tok,
                enable_constrained_by_default=False,
                sync_cuda=True,
            )
            kg_decode_time += decode_time
            kg_decode_per_constraint[cid] = decode_time
            num_kg_decode_calls += 1

            if raw is None:
                continue
            if isinstance(raw, str):
                raw = [raw]
            paths_per_constraint[cid] = raw

    # ── Step 3: Intersect ────────────────────────────────────────────────────
    raw_final_candidates, intersect_time = timed_call(
        intersector.intersect,
        paths_per_constraint,
        filter_constraints=filter_constraints,
    )
    intersect_stats = getattr(intersector, "last_stats", {})

    answer_refine_stats = {}
    if args.answer_aware:
        final_candidates, answer_refine_stats = refine_candidates(
            question=question,
            data=data,
            decomposed=decomposed,
            candidates=raw_final_candidates,
            max_hops=args.answer_expand_hops,
            max_added=args.answer_expand_max_added,
            undirected=args.undirected,
        )
    else:
        final_candidates = raw_final_candidates
        answer_refine_stats = {
            "used_answer_aware": False,
            "original_size": len(raw_final_candidates),
            "final_size": len(raw_final_candidates),
        }

    evidence_paths, evidence_collect_time = timed_call(
        intersector.collect_evidence_paths,
        paths_per_constraint, raw_final_candidates, max_paths=args.evidence_paths_per_constraint
    )
    # Fallback evidence when intersection is empty
    if not evidence_paths:
        evidence_fallback_start = perf_counter()
        for paths in paths_per_constraint.values():
            evidence_paths.extend(paths[:2])
        evidence_collect_time += elapsed_since(evidence_fallback_start)

    evidence_format_start = perf_counter()
    evidence_str = format_paths_for_display(
        paths_per_constraint,
        max_per=args.evidence_paths_per_constraint,
        extra_paths=answer_refine_stats.get("expansion_paths", []),
    )
    evidence_format_time = elapsed_since(evidence_format_start)

    # ── Step 4: Final answer ─────────────────────────────────────────────────
    final_prompt_start = perf_counter()
    if final_candidates:
        candidates_str = "\n".join(sorted(final_candidates))
        final_prompt   = FINAL_ANSWER_WITH_CANDIDATES.format(
            question=question,
            evidence_paths=evidence_str,
            candidates=candidates_str,
        )
    else:
        final_prompt = FINAL_ANSWER_NO_CANDIDATES.format(
            question=question,
            evidence_paths=evidence_str,
        )

    final_input = general_model.prepare_model_prompt(final_prompt)
    final_prompt_build_time = elapsed_since(final_prompt_start)
    prediction, final_api_time = timed_call(general_model.generate_sentence, final_input)
    if prediction is None:
        return None

    pred_list = parse_answer_lines(prediction)
    total_time = elapsed_since(sample_start, sync_cuda=True)
    decompose_api_calls = int(decompose_meta.get("api_calls", 0) or 0)
    num_api_calls = decompose_api_calls + 1

    return {
        "id":           qid,
        "question":     question,
        "prediction":   pred_list,
        "ground_truth": answer,
        "input":        final_input,
        "inference_time": total_time,
        "timing": {
            "schema_version": 1,
            "unit": "seconds",
            "method": "dvi_path_scorer" if args.path_scorer else "dvi",
            "total": total_time,
            "decompose_time": decompose_time,
            "decompose_api_time": decompose_time if decompose_api_calls > 0 else 0.0,
            "decompose_cache_hit": bool(decompose_meta.get("cache_hit", False)),
            "decompose_api_calls": decompose_api_calls,
            "trie_build_time": trie_build_time,
            "constraint_prompt_build_time": constraint_prompt_build_time,
            "kg_decode_time": kg_decode_time,
            "kg_decode_per_constraint": kg_decode_per_constraint,
            "num_kg_decode_calls": num_kg_decode_calls,
            "path_scorer_time": path_scorer_time,
            "intersect_time": intersect_time,
            "evidence_collect_time": evidence_collect_time,
            "evidence_format_time": evidence_format_time,
            "final_prompt_build_time": final_prompt_build_time,
            "final_api_time": final_api_time,
            "num_api_calls": num_api_calls,
            "num_final_candidates": len(final_candidates),
            "num_raw_final_candidates": len(raw_final_candidates),
            "num_evidence_paths": len(evidence_paths),
            "kg_model_name": args.kg_model_name,
            "general_model_name": args.general_model_name,
        },
        "dvi_meta": {
            "mode":                     "path_scorer" if args.path_scorer else "kg_llm",
            "constraints":              decomposed.get("constraints", []),
            "n_relation_constraints":   len(rel_constraints),
            "n_filter_constraints":     len(filter_constraints),
            "paths_sizes":              {k: len(v) for k, v in paths_per_constraint.items()},
            "trie_stats":               trie_stats,
            "intersect_stats":          intersect_stats,
            "answer_refine_stats":      answer_refine_stats,
            "raw_final_candidates":     sorted(raw_final_candidates),
            "final_candidates":         sorted(final_candidates),
            "selection_strategy":       selection_strategy,
            "used_fallback":            "c_fallback" in paths_per_constraint or selection_strategy != "per_constraint",
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Dataset
    dataset = load_dataset(os.path.join(args.data_path, args.d), split=args.split)

    # Output dir
    if args.path_scorer:
        method_dir = f"DVI-Scorer-x-{args.general_model_name}"
        mode_tag = (
            f"scorer-bik{args.bi_k}-crossk{args.cross_k}-hop{args.index_path_length}"
            + (f"-{args.variant_name}" if args.variant_name else "")
        )
    else:
        method_dir = f"DVI-{args.kg_model_name}-x-{args.general_model_name}"
        mode_tag = (
            f"k{args.k}-hop{args.index_path_length}"
            + (f"-{args.variant_name}" if args.variant_name else "")
        )
    output_dir = os.path.join(args.predict_path, args.d, method_dir, args.split, mode_tag)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.decompose_cache_path), exist_ok=True)
    print(f"[DVI] Output → {output_dir}")

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load verifier ───────────────────────────────────────────────────────
    kg_model = None
    path_scorer = None
    tokenizer_for_builder = None
    if args.path_scorer:
        from src.dvi import PathScorer

        print("[DVI] Loading PathScorer verifier")
        print(f"  bi-encoder:    {args.bi_encoder}")
        print(f"  cross-encoder: {args.cross_encoder}")
        print(f"  bi_k={args.bi_k} cross_k={args.cross_k}")
        path_scorer = PathScorer(
            bi_encoder_name=args.bi_encoder,
            cross_encoder_name=args.cross_encoder,
            bi_k=args.bi_k,
            cross_k=args.cross_k,
            device=args.path_scorer_device,
        )
    else:
        print(f"[DVI] Loading KG model: {args.kg_model_name}")
        kg_args      = make_kg_args(args)
        KGModelClass = get_registed_model(kg_args.model_name)
        kg_model     = KGModelClass(kg_args)
        kg_model.prepare_for_inference()
        tokenizer_for_builder = kg_model.tokenizer

    # ── Load General LLM (ChatGPT / gpt-4o-mini) ────────────────────────────
    print(f"[DVI] Loading General model: {args.general_model_name}")
    gen_args        = make_general_args(args)
    GenModelClass   = get_registed_model(gen_args.model_name)
    general_model   = GenModelClass(gen_args)
    general_model.prepare_for_inference()

    # ── DVI components ───────────────────────────────────────────────────────
    prompt_builder = DVIPromptBuilder(
        tokenizer=tokenizer_for_builder,
        prompt="zero-shot",
        index_path_length=args.index_path_length,
        undirected=args.undirected,
    )
    decomposer = QueryDecomposer(
        model_name=args.general_model_name,
        retry=args.retry,
        cache=True,
    )
    if args.decompose_cache_path and os.path.exists(args.decompose_cache_path):
        decomposer.load_cache(args.decompose_cache_path)

    intersector = CandidateIntersector(
        min_candidates=args.min_candidates,
        verbose=args.debug,
    )

    # ── Inference loop ───────────────────────────────────────────────────────
    pred_file = os.path.join(output_dir, "predictions.jsonl")
    fout, processed_list = get_output_file(pred_file, force=args.force)

    for data in tqdm(dataset, desc="[DVI]"):
        if data["id"] in processed_list:
            continue
        try:
            result = run_dvi_on_sample(
                data, kg_model, general_model,
                path_scorer,
                prompt_builder, decomposer, intersector, args,
            )
        except Exception as e:
            print(f"[DVI] Error on {data['id']}: {e}")
            result = None

        if result:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            if args.debug:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[DVI] Skipped: {data['id']}")

    fout.close()

    # Save decompose cache
    decomposer.save_cache(args.decompose_cache_path)
    print(f"[DVI] Decompose cache → {args.decompose_cache_path}")

    # Evaluate
    write_timing_summary(pred_file)
    print("[DVI] Evaluating …")
    eval_result(pred_file)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_path",     default="rmanluo")
    p.add_argument("--d", "-d",       default="RoG-cwq")
    p.add_argument("--split",         default="test")
    p.add_argument("--predict_path",  default="results/DVI")

    # KG-specialized LLM
    p.add_argument("--kg_model_name", default="GCR-Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--kg_model_path", default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")

    # General LLM
    p.add_argument("--general_model_name", default="gpt-4o-mini")

    # Decoding
    p.add_argument("--k",                  type=int, default=10)
    p.add_argument("--index_path_length",  type=int, default=2)
    p.add_argument("--generation_mode",    default="group-beam")
    p.add_argument("--undirected",         type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--attn_implementation",default="flash_attention_2")
    p.add_argument("--dtype",              default="bf16")
    p.add_argument("--quant",              default="none")
    p.add_argument("--max_new_tokens",     type=int, default=512)
    p.add_argument("--maximun_token",      type=int, default=4096)
    p.add_argument("--chat_model",         type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--use_assistant_model",type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--assistant_model_path", default=None)

    # PathScorer verifier (v2)
    p.add_argument("--path_scorer", action="store_true",
                   help="Use DFS path enumeration + bi/cross encoder ranking instead of KG-LLM decoding")
    p.add_argument("--bi_encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-transformers bi-encoder used for coarse path retrieval")
    p.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                   help="Cross-encoder used for path reranking")
    p.add_argument("--bi_k", type=int, default=100,
                   help="Number of paths kept after bi-encoder retrieval")
    p.add_argument("--cross_k", type=int, default=10,
                   help="Number of paths kept after cross-encoder reranking")
    p.add_argument("--path_scorer_device", default=None,
                   help="PathScorer device override: cpu, cuda, or unset for auto")

    # DVI
    p.add_argument("--min_candidates",       type=int, default=1)
    p.add_argument("--decompose_cache_path", default="data/decompose_cache.json")
    p.add_argument("--variant_name",         default="")
    p.add_argument("--answer_aware",         type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--answer_expand_hops",   type=int, default=2)
    p.add_argument("--answer_expand_max_added", type=int, default=12)
    p.add_argument("--selective_fallback",   type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--evidence_paths_per_constraint", type=int, default=3)

    # Misc
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    main(args)
