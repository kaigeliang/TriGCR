# Results Summary

This file summarizes the main metrics. For the full table set, see [`EXPERIMENT_TABLES.md`](EXPERIMENT_TABLES.md).

## Main Full-Test Results

| System | Dataset / Setting | Acc./Rec. | Hit | Hit@1 | F1 | Time / sample |
|---|---:|---:|---:|---:|---:|---:|
| GCR + Llama-3.1-8B | RoG-CWQ full test | 59.01 | 64.34 | 55.88 | 54.55 | baseline |
| GCR + Llama-2-7B | RoG-CWQ full test | 60.51 | 66.07 | 57.12 | 54.50 | baseline |
| GCR + Qwen2-0.5B | RoG-CWQ full test | -- | -- | -- | 42.31 | baseline |
| Routed DVI + Llama-3.1-8B | RoG-CWQ full test, <code>&#124;C&#124;=2</code> gate | 59.26 | -- | 56.22 | 54.63 | 6.58s |
| Routed DVI + Llama-2-7B | RoG-CWQ full test, <code>&#124;C&#124;=2</code> gate | 60.56 | -- | 56.92 | 54.51 | 6.53s |
| GraphLite / PathScorer | RoG-CWQ aligned scorer run | -- | -- | -- | 42.14 | 3.09s |
| Embedding-guided KG-Trie | CWQ20, 3-hop embedding top-1 | -- | 55.00 | -- | 45.56 | probe |

## DVI Gate Ablation

| Model | Gate | Acc./Rec. | Hit@1 | F1 | Time / sample |
|---|---:|---:|---:|---:|---:|
| Llama-3.1 | <code>&#124;C&#124; <= 1</code> | 59.03 | 54.80 | 54.33 | 8.54s |
| Llama-3.1 | <code>&#124;C&#124; = 2</code> | 59.26 | 56.22 | 54.63 | 6.58s |
| Llama-3.1 | <code>&#124;C&#124; <= 2</code> | 59.28 | 55.14 | 54.41 | 10.29s |
| Llama-3.1 | <code>&#124;C&#124; <= 3</code> | 58.92 | 54.63 | 53.87 | 11.27s |
| Llama-2 | <code>&#124;C&#124; <= 1</code> | 60.40 | 55.88 | 54.38 | 6.89s |
| Llama-2 | <code>&#124;C&#124; = 2</code> | 60.56 | 56.92 | 54.51 | 6.53s |
| Llama-2 | <code>&#124;C&#124; <= 2</code> | 60.45 | 55.68 | 54.40 | 7.16s |
| Llama-2 | <code>&#124;C&#124; <= 3</code> | 59.71 | 54.91 | 53.84 | 7.34s |

## GraphLite / PathScorer Training Signal

| Verifier | Groups | MRR | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|
| Pretrained cross-encoder | 158 | 68.04 | 56.33 | 85.44 | 88.61 |
| Fine-tuned cross-encoder | 158 | 77.13 | 69.62 | 87.97 | 91.77 |

The fine-tuned scorer improves path-level retrieval, but final QA remains below the strongest GCR/DVI systems, which is why the report treats GraphLite as a fast verifier/prefilter rather than a standalone replacement.

## Embedding-Guided Hop-Length Ablation

| CWQ20 setting | Hit | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| 2-hop group-beam baseline | 45.00 | 27.83 | 30.00 | 36.88 |
| 2-hop beam early stopping | 45.00 | 30.50 | 34.17 | 36.88 |
| 3-hop budgeted, no embedding | 45.00 | 22.37 | 20.83 | 31.29 |
| 3-hop embedding, all candidates | 60.00 | 37.00 | 37.50 | 51.88 |
| 3-hop embedding, top-1 | 55.00 | 45.56 | 55.00 | 43.79 |

## Interpretation

The three extensions address different failure modes. DVI targets aggregation and constraint intersection, GraphLite targets verification cost and path scoring, and embedding-guided KG-Trie targets retrieval coverage for longer-hop evidence. The strongest practical result in this archive is selective DVI routing, while the strongest research signal is that the three methods are complementary rather than mutually exclusive.
