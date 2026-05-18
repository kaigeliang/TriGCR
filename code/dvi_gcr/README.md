# GCR-DVI: Decompose-Verify-Intersect for Knowledge-Grounded LLM Reasoning

**AIAA 4051 Final Research Project**  
Project: *Knowledge Grounded LLM Reasoner*  
Method: **DVI (Decompose-Verify-Intersect)**  
Base code: [Graph-Constrained Reasoning (GCR)](https://github.com/RManLuo/graph-constrained-reasoning) · ICML 2025

---

## Overview

GCR uses a KG-Trie to constrain LLM decoding so that generated reasoning paths stay faithful to the knowledge graph. However, **KG-Trie only encodes linear prefix paths**, causing failures on compositional queries with multiple constraints.

### The Problem

For a query like:
> *"Which American actors born before 1970 have won both a Golden Globe and an Oscar, and starred in films directed by Christopher Nolan?"*

GCR builds separate KG-Tries for each entity (Nolan, Oscar, Golden Globe) and asks the LLM to mentally find the intersection — causing hallucination on `conjunction`-type queries.

### Our Solution: DVI

```
Query
  │
  ▼  Step 1 · Decompose  (General LLM: gpt-4o-mini)
Atomic Constraints
  c1: Nolan → films → ?actor
  c2: ?actor → won → Oscar
  c3: ?actor → won → Golden Globe
  │
  ▼  Step 2 · Verify  (KG-LLM + per-constraint mini-Trie)
Candidate Sets
  S1 = {DiCaprio, Caine, Murphy, ...}
  S2 = {DiCaprio, Hanks, Streep, ...}
  S3 = {DiCaprio, Washington, ...}
  │
  ▼  Step 3 · Intersect  (Python set.intersection — zero hallucination)
Final Candidates = S1 ∩ S2 ∩ S3 = {DiCaprio}
  │
  ▼  Step 4 · Answer  (General LLM)
"Leonardo DiCaprio"
```

**Key insight**: Move intersection from the LLM to Python → zero hallucination on `conjunction`-type queries.

---

## Project Structure

```
dvi_gcr/
├── src/
│   ├── dvi/
│   │   ├── __init__.py              # Module exports
│   │   ├── decomposer.py            # Step 1: Query → atomic constraints (LLM)
│   │   └── intersector.py           # Step 3: Programmatic set intersection
│   ├── qa_prompt_builder.py         # + DVIPromptBuilder (per-constraint Tries)
│   ├── graph_constrained_decoding.py
│   ├── trie.py
│   └── llms/
│       ├── chatgpt.py
│       ├── graph_constrained_decoding_model.py
│       └── ...
├── workflow/
│   ├── predict_paths_and_answers.py  # GCR baseline Step 1
│   ├── predict_final_answer.py       # GCR baseline Step 2
│   ├── predict_dvi.py                # DVI end-to-end pipeline   ← NEW
│   ├── failure_analysis.py           # Per-type failure analysis  ← NEW
│   └── build_graph_index.py
├── scripts/
│   ├── run_baseline_cwq.sh           # Reproduce baseline        ← NEW
│   ├── run_dvi_cwq.sh                # Run DVI                   ← NEW
│   └── run_failure_analysis.sh       # Run analysis              ← NEW
└── .env                              # API keys (copy from .env.example)
```

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n GCR python=3.12
conda activate GCR
pip install poetry
poetry install
pip install flash-attn --no-build-isolation   # faster inference
```

### 2. API Keys

```bash
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=<your_openai_key>
#   HF_TOKEN=<your_huggingface_token>        (needed for Llama models)
```

### 3. Run Everything (End-to-End)

```bash
conda activate GCR

# Part 1: Reproduce baseline with 2 pre-trained models (~9h GPU)
bash scripts/run_baseline_cwq.sh

# Part 2a: Failure analysis on baseline (~5 min)
bash scripts/run_failure_analysis.sh

# Part 2b: Run DVI (~10h GPU)
bash scripts/run_dvi_cwq.sh

# Part 2c: Compare DVI vs Baseline by compositionality_type
bash scripts/run_failure_analysis.sh   # re-run to get delta comparison
```

---

## Pre-trained Models

All models **auto-download** from HuggingFace — no training required.

| Model | HuggingFace Path | GPU RAM |
|---|---|---|
| Llama-3.1-8B | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | ~16 GB |
| Qwen2-0.5B | `rmanluo/GCR-Qwen2-0.5B-Instruct` | ~2 GB |
| Llama-2-7B | `rmanluo/GCR-Llama-2-7b-chat-hf` | ~14 GB |

---

## Detailed CLI Reference

### Baseline

```bash
# Step 1: KG-specialized LLM generates reasoning paths
python workflow/predict_paths_and_answers.py \
  --data_path rmanluo --d RoG-cwq --split test \
  --model_name GCR-Meta-Llama-3.1-8B-Instruct \
  --model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --k 10 --index_path_length 2 \
  --prompt_mode zero-shot --generation_mode group-beam

# Step 2: General LLM aggregates paths into final answer
python workflow/predict_final_answer.py \
  --data_path rmanluo --d RoG-cwq --split test \
  --model_name gpt-4o-mini \
  --reasoning_path results/GenPaths/RoG-cwq/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10-index_len2/predictions.jsonl \
  --add_path True -n 8
```

### DVI

```bash
python workflow/predict_dvi.py \
  --data_path rmanluo --d RoG-cwq --split test \
  --kg_model_name GCR-Meta-Llama-3.1-8B-Instruct \
  --kg_model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --general_model_name gpt-4o-mini \
  --k 10 --index_path_length 2 \
  --decompose_cache_path data/decompose_cache.json \
  --min_candidates 1 \
  [--debug]
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--k` | 10 | Beam width per constraint |
| `--index_path_length` | 2 | Max KG hops for Trie |
| `--min_candidates` | 1 | Relax intersection if empty |
| `--decompose_cache_path` | `data/decompose_cache.json` | Cache to save API calls |

### Failure Analysis

```bash
python workflow/failure_analysis.py \
  --pred_file results/DVI/.../predictions.jsonl \
  --baseline_pred_file results/GenPaths/.../predictions.jsonl \
  --data_path rmanluo --d RoG-cwq --split test \
  --output_dir results/failure_analysis/dvi_vs_baseline \
  --n_cases 10
```

Output:
```
Type                      N   Hit@1       F1
----------------------------------------------
conjunction             521  0.xxxx   0.xxxx   (Δhit +0.xxxx, Δf1 +0.xxxx)
composition             438  0.xxxx   0.xxxx
comparative              87  0.xxxx   0.xxxx
superlative              54  0.xxxx   0.xxxx
----------------------------------------------
Overall                1100  0.xxxx   0.xxxx
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Hit@1** | Is the top prediction in the answer set? |
| **F1** | Harmonic mean of precision and recall |
| **Precision** | Fraction of predictions that are correct |
| **Recall** | Fraction of correct answers that are predicted |

---

## Experiment Plan

| ID | Method | KG Model | CWQ Hit@1 | CWQ F1 | Note |
|---|---|---|---|---|---|
| E1 | GCR Baseline | Llama-3.1-8B | — | — | Part 1 (20%) |
| E2 | GCR Baseline | Qwen2-0.5B | — | — | Part 1 (10%) |
| E3 | **DVI (ours)** | Llama-3.1-8B | — | — | Part 2 main |
| E4 | DVI (ours) | Qwen2-0.5B | — | — | Part 2 ablation |

### Ablation Study

| Variant | Decompose | Per-Constraint Trie | Intersect |
|---|---|---|---|
| E1 Baseline | ✗ | ✗ | ✗ |
| DVI no-intersect | ✓ | ✓ | ✗ (LLM aggregates) |
| **DVI full (E3)** | ✓ | ✓ | ✓ |

---

## How DVI Modifies the Graph Constrained Decoding Module

Per the project requirement: *"Modify graph constrained decoding module to handle complex reasoning constraints."*

| Component | GCR Baseline | DVI |
|---|---|---|
| **Trie construction** | One global Trie from ALL entity DFS paths | Per-constraint mini-Trie from single anchor |
| **Decoding** | One beam search, all entities mixed | Separate beam search per constraint |
| **Aggregation** | LLM reads paths and guesses intersection (hallucination) | `set.intersection()` — deterministic |
| **Prompt** | All entities listed as topic entities | Per-constraint prompt with anchor + natural language hint |

---

## Expected GPU Hours

| Task | Model | Est. Hours |
|---|---|---|
| Baseline (Llama-3.1-8B on CWQ test) | 8B | ~8h |
| Baseline (Qwen2-0.5B on CWQ test) | 0.5B | ~1h |
| DVI (Llama-3.1-8B on CWQ test) | 8B | ~10h |
| **Total** | | **~20h** (within 50h budget) |

---

## Related Work

1. Luo et al. *Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models.* ICML 2025.
2. Talmor & Berant. *The Web as a Knowledge-base for Answering Complex Questions.* NAACL 2018. (CWQ Dataset)

---

## Team

AIAA 4051 · HKUST(GZ) · Spring 2026
