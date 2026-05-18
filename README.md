# TriGCR: Constraint-Aware Graph Reasoning for Faithful Knowledge-Grounded QA

**TriGCR** studies faithful knowledge-grounded question answering with graph-constrained reasoning. The code builds on Graph-Constrained Reasoning (GCR) and adds three extensions for complex KGQA:

1. **DVI (Decompose-Verify-Intersect)**: decomposes a complex question into atomic constraints, verifies each constraint on the knowledge graph, and intersects candidate answer sets with deterministic Python logic.
2. **GraphLite / PathScorer**: replaces expensive KG-LLM verification with explicit graph enumeration plus neural path/entity scoring.
3. **Embedding-guided KG-Trie**: ranks candidate paths by semantic similarity before trie construction so that useful 3-hop evidence can be added without uncontrolled path explosion.

## Visual Overview

<p align="center">
  <img src="figures/trigcr_project_overview.png" alt="TriGCR project overview" width="900">
</p>

<p align="center"><strong>Project overview:</strong> the three research tracks extend the reproduced GCR baseline from complementary directions. DVI adds explicit constraint state and intersection, GraphLite studies a faster verifier, and embedding-guided KG-Trie improves long-hop evidence selection.</p>

<p align="center">
  <img src="figures/trigcr_method_architecture.png" alt="TriGCR method architecture" width="900">
</p>

<p align="center"><strong>Method architecture:</strong> implementation-level comparison of DVI, GraphLite/PathScorer, and embedding-guided KG-Trie construction.</p>

<p align="center">
  <img src="figures/trigcr_reasoning_traces.png" alt="TriGCR three-method reasoning traces" width="900">
</p>

<p align="center"><strong>Question-to-answer traces:</strong> how the three methods process the same KGQA problem through decomposition, verification, path scoring, candidate construction, and final answering.</p>


## Key Results

| System | Setting | F1 | Hit@1 | Time / sample |
|---|---|---:|---:|---:|
| GCR baseline + Llama-3.1-8B | RoG-CWQ full test | 54.55 | 55.88 | 2.06s |
| Routed DVI + Llama-3.1-8B | <code>&#124;C&#124;=2</code> gate | **54.63** | **56.22** | 6.58s |
| GCR baseline + Llama-2-7B | RoG-CWQ full test | 54.50 | **57.12** | 6.27s |
| Routed DVI + Llama-2-7B | <code>&#124;C&#124;=2</code> gate | **54.51** | 56.92 | 6.53s |
| GraphLite / PathScorer | Entity/path verifier | 42.14 | 45.09 | **3.09s** |
| Embedding-guided KG-Trie | CWQ20, 3-hop embedding top-1 | 45.56 | -- | probe |

Experiment tables, ablations, oracle-routing headroom, and representative failure cases are in [`results_summary/EXPERIMENT_TABLES.md`](results_summary/EXPERIMENT_TABLES.md).

## Repository Layout

```text
TriGCR/
├── code/
│   ├── dvi_gcr/                 # Main GCR baseline + DVI + PathScorer integration
│   ├── embedding_guided_kgtrie/        # Overlay patch for embedding-guided KG-Trie construction
│   └── graphlite/               # GraphLite prototype, extracted from lite_framework.rar
├── report/
│   └── TriGCR_final_report.pdf
├── figures/                    # Overview, architecture, and trace figures used in README/report
├── results_summary/
│   ├── RESULTS.md               # Compact metric summary from archived experiments
│   └── EXPERIMENT_TABLES.md      # Main tables and ablations
├── datasets/
│   └── DATASET_LINKS.md          # Dataset URLs and loading commands
├── poster/
│   └── TriGCR_poster.pdf
├── docs/
│   └── assignment_requirements.pdf
├── environment_GCR.yml
├── requirements_GCR.txt
├── LICENSE
└── README.md
```

## Main Code Entry Points

| Method | Files |
|---|---|
| GCR baseline | `code/dvi_gcr/workflow/predict_paths_and_answers.py`, `code/dvi_gcr/workflow/predict_final_answer.py`, `code/dvi_gcr/src/graph_constrained_decoding.py`, `code/dvi_gcr/src/trie.py` |
| DVI | `code/dvi_gcr/workflow/predict_dvi.py`, `code/dvi_gcr/src/dvi/decomposer.py`, `code/dvi_gcr/src/dvi/intersector.py`, `code/dvi_gcr/src/dvi/answer_aware.py`, `code/dvi_gcr/src/qa_prompt_builder.py` |
| GraphLite / PathScorer | `code/graphlite/lite_framework/entity_level_decoder.py`, `code/graphlite/lite_framework/cross_encoder_verifier.py`, `code/dvi_gcr/src/dvi/path_scorer.py`, `code/dvi_gcr/workflow/build_path_verifier_data.py`, `code/dvi_gcr/workflow/train_path_reranker.py` |
| Embedding-guided KG-Trie | `code/embedding_guided_kgtrie/overlay/src/qa_prompt_builder.py`, `code/embedding_guided_kgtrie/overlay/workflow/predict_paths_and_answers.py`, `code/embedding_guided_kgtrie/kgtrie_enhanced_changes.patch` |

## Setup

The original environment was developed with Python 3.12 and the GCR dependency stack. Either install from the root requirement files:

```bash
conda env create -f environment_GCR.yml
conda activate GCR
pip install -r requirements_GCR.txt
```

or use the Poetry project under `code/dvi_gcr/`:

```bash
cd code/dvi_gcr
pip install poetry
poetry install
```

Create a local `.env` file from the provided example when running API-backed final-answer judging or decomposition:

```bash
cp code/dvi_gcr/.env.example code/dvi_gcr/.env
```

Do not commit `.env`, model checkpoints, generated prediction files, or raw datasets.

## Quick Checks

After activating the `GCR` environment, these commands check imports and command-line entry points without running full model inference.

```bash
python -m compileall -q code/dvi_gcr code/graphlite code/embedding_guided_kgtrie

cd code/dvi_gcr
python workflow/predict_paths_and_answers.py --help
python workflow/predict_dvi.py --help
python workflow/predict_final_answer.py --help
python workflow/build_path_verifier_data.py --help
python workflow/train_path_reranker.py --help

cd ../embedding_guided_kgtrie/overlay
python workflow/predict_paths_and_answers.py --help

cd ../../graphlite/lite_framework
python entity_level_predict.py --help
python train_cross_encoder.py --help
python train_entity_verifier.py --help
```

Full experiments require the Hugging Face model checkpoints, the RoG datasets listed in `datasets/DATASET_LINKS.md`, and API credentials for general-LLM decomposition or final answering.

## Example Commands

Run the GCR-style path generation stage:

```bash
cd code/dvi_gcr
python workflow/predict_paths_and_answers.py \
  --data_path rmanluo \
  --d RoG-cwq \
  --split test \
  --model_name GCR-Meta-Llama-3.1-8B-Instruct \
  --model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --k 10 \
  --index_path_length 2 \
  --prompt_mode zero-shot \
  --generation_mode group-beam
```

Run DVI:

```bash
cd code/dvi_gcr
python workflow/predict_dvi.py \
  --data_path rmanluo \
  --d RoG-cwq \
  --split test \
  --kg_model_name GCR-Meta-Llama-3.1-8B-Instruct \
  --kg_model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --general_model_name gpt-4o-mini \
  --k 10 \
  --index_path_length 2 \
  --decompose_cache_path data/decompose_cache.json \
  --min_candidates 1
```

Train the integrated path reranker used by GraphLite-style verification:

```bash
cd code/dvi_gcr
python workflow/build_path_verifier_data.py --help
python workflow/train_path_reranker.py --help
```

Apply or inspect the embedding-guided KG-Trie overlay:

```bash
cd code/embedding_guided_kgtrie
less kgtrie_enhanced_changes.patch
```

## Results and Report

The paper is at `report/TriGCR_final_report.pdf`, and the poster is at `poster/TriGCR_poster.pdf`. Dataset links are listed in `datasets/DATASET_LINKS.md`. A concise metric summary is in `results_summary/RESULTS.md`.

## Provenance

This work builds on the MIT-licensed Graph-Constrained Reasoning codebase. The included `LICENSE` is copied from the base project.
