# Lite Framework: Entity-Level Decoder for KGQA

## Motivation

GCR (Graph-Constrained Reasoning) achieves strong accuracy (CWQ ~77%) by constraining LLM beam search through a knowledge graph trie. But it has fundamental limitations:

1. **Slow inference.** Beam search runs 256 token-generation steps × 10 beams, each requiring a full LLM forward pass. ~23s/sample.

2. **Beam score ≠ correctness.** Beam search selects tokens by cumulative log-prob, which measures language fluency, not answer correctness. `common.topic.image` outranks `law.inventor.inventions` simply because it's more frequent in training data.

3. **Signal injection is broken.** All attempts to bias beam search (PPR, embedding similarity, keyword matching, hidden-state similarity) failed to improve accuracy. The root cause: per-token bias is too weak to overcome the model's log-P preferences, and 88% of trie branch points have zero differentiation signal.

The Entity-Level Decoder (Lite) asks: **what if we replace token-level generation with structured graph search?**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  Stage 1: Relation Selection                              │
│  ─────────────────────────                                │
│  Enumerate 45 first-hop relations from topic entities.    │
│  DeBERTa cross-encoder (184M, trained on 100 questions)   │
│  scores each (question, relation) pair → top-K.           │
│                                                           │
│  Why DeBERTa? Relation relevance is a semantic matching   │
│  task. "inventions" relates to "invent"; "gender" doesn't.│
│  No factual knowledge needed. 100 training questions      │
│  sufficient. Cross-attention lets question tokens directly │
│  attend relation tokens.                                  │
│                                                           │
│  Stage 2: Entity Scoring                                  │
│  ──────────────────────                                    │
│  For each selected relation, enumerate all 1-hop entities. │
│  For Freebase MID entities (m.xxxxx), expand to 2-hop to  │
│  expose human-readable entities.                          │
│                                                           │
│  Entity scoring: trained entity verifier (DeBERTa, 184M)  │
│  scores (question, path, entity) triples → [0,1].         │
│                                                           │
│  Stage 3: Output                                          │
│  ─────────────                                            │
│  Top-scored entities returned as answers with KG paths.   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Comparison with GCR

| | GCR | Lite |
|------|-----|------|
| Search mechanism | Token-by-token beam search | Structured graph enumeration + batch scoring |
| Forward passes | ~256 per sample | ~15-30 per sample |
| Inference speed | ~23s/sample | ~3s/sample (GPU) |
| Scoring signal | Beam log-P (fluency) | Trained verifier (correctness) |
| Relation selection | Implicit in beam search | Explicit DeBERTa cross-encoder |
| Entity selection | Beam search dynamics | Trained entity verifier |

## Results

### CWQ Full Test Set (3531 samples)

Evaluated by GPT-4o-mini judge (replicating original paper's evaluation protocol):

```
Accuracy:  54.66%
Hit:       59.25%
F1:        44.90%
Precision: 43.70%
Recall:    54.66%
```

Per-sample inference time: ~1.5s (GPU, Llama 8B + DeBERTa 184M).

### WebQSP test[:50]

| | Accuracy | Hit | Speed |
|------|----------|-----|-------|
| GCR baseline | 76.54% | 86.0% | 26.4s |
| Lite (k=15) | 78.42% | 82.0% | 8.6s |
| Lite (k=50) | **83.58%** | **88.0%** | 25.2s |

### Ablation: CWQ test[:50]

| Configuration | Accuracy |
|------|----------|
| Perplexity only (no verifier) | ~48% |
| + Relation verifier | ~48% |
| + Entity verifier | ~48% |
| + 2-hop expansion (all entities) | **69.07%** |

Key insight: 68.8% of CWQ correct answers require 2-hop through non-MID intermediate entities. Early versions only expanded 2-hop from Freebase MIDs, creating an artificial ceiling.

## Limitations and Future Work

### Current Limitations

1. **Entity verifier trained on only 100 questions.** Full training data (30K questions) not yet utilized. Expected significant gain.

2. **DeBERTa lacks factual knowledge.** Cannot distinguish "Vladimir Lenin" from "Joseph Stalin" as the first Soviet dictator. Llama has this knowledge.

3. **No re-ranking stage.** Output is sorted by entity verifier score. An LLM judge stage could further improve accuracy.

4. **CWQ multi-entity complexity.** 48% of CWQ questions have multiple topic entities, requiring the verifier to understand cross-entity constraints.

### Future Directions

1. **Full training of entity verifier.** Use all 30K CWQ + WebQSP training questions. Expected accuracy gain: 5-15pp.

2. **Llama-based entity scoring.** Fine-tune Llama 8B with LoRA for entity-level binary classification. Leverages pre-trained factual knowledge about entities.

3. **LLM judge re-ranking.** Add a final stage where Llama reads the top-k paths and selects the best answer. One additional forward pass.

4. **KV cache sharing.** All entity prefixes share the same prompt. Caching prompt KV and reusing for path tokens would accelerate inference ~3x.

5. **Better 2-hop pruning.** Current approach expands 2-hop from all entities above threshold. A trained scorer could selectively expand only promising paths.

## Core Files

| File | Purpose |
|------|---------|
| `entity_level_decoder.py` | Main decode logic: graph enumeration, relation/entity scoring, 2-hop expansion |
| `cross_encoder_verifier.py` | DeBERTa-based verifier: binary classification for relation/entity relevance |
| `entity_level_predict.py` | Inference pipeline: dataset loading, decoder invocation, output formatting |
| `train_cross_encoder.py` | Training script for relation verifier (question, relation) → score |
| `train_entity_verifier.py` | Training script for entity verifier (question, path, entity) → score |
