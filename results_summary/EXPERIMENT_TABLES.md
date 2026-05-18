# Experiment Tables

This file collects the main experiment tables from the TriGCR report. Values are percentages unless noted otherwise.

## Table 1: Full-Test RoG-CWQ Results

| Method | KG / verifier | N | Acc./Rec. | Hit | Hit@1 | F1 | Prec. | Time(s) | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GCR baseline | Llama-3.1-8B | 3531 | 59.01 | 64.34 | 55.88 | 54.55 | 54.87 | 2.06 | 882/3531 |
| GCR baseline | Llama-2-7B | 3531 | 60.51 | 66.07 | **57.12** | 54.50 | 54.29 | 6.27 | 3531/3531 |
| GCR baseline | Qwen2-0.5B | 3531 | 47.81 | 54.46 | 44.83 | 42.31 | 42.52 | 1.68 | 3531/3531 |
| Raw DVI | Llama-3.1-8B | 3520 | 57.71 | 64.32 | 54.18 | 52.57 | 53.42 | 12.72 | 3520/3520 |
| Routed DVI, <code>&#124;C&#124;=2</code> | Llama-3.1-8B | 3531 | 59.26 | 64.68 | 56.22 | **54.63** | **54.90** | 6.58 | 1307/3531 |
| Raw DVI | Llama-2-7B | 3520 | 57.76 | 64.35 | 53.86 | 52.31 | 53.02 | 7.92 | 3520/3520 |
| Routed DVI, <code>&#124;C&#124;=2</code> | Llama-2-7B | 3531 | **60.56** | **66.27** | 56.92 | 54.51 | 54.42 | 6.53 | 3531/3531 |
| GraphLite / PathScorer | Entity / path verifier | 3520 | 51.05 | 56.19 | 45.09 | 42.14 | 40.64 | **3.09** | 3520/3520 |

Main takeaway: routed DVI gives small full-test gains over the strongest reproduced GCR baselines, while GraphLite gives the fastest modified verifier but lower QA quality.

## Table 2: Post-Hoc Oracle Routing Headroom

| KG model | Source | N | Hit@1 | F1 | Hit |
|---|---|---:|---:|---:|---:|
| Llama-3.1 | GCR | 3520 | 56.05 | 54.63 | 64.43 |
| Llama-3.1 | Raw DVI | 3520 | 54.18 | 52.57 | 64.32 |
| Llama-3.1 | Oracle | 3520 | **62.36** | **61.51** | **71.45** |
| Llama-2 | GCR | 3520 | 57.27 | 54.57 | 66.16 |
| Llama-2 | Raw DVI | 3520 | 53.86 | 52.31 | 64.35 |
| Llama-2 | Oracle | 3520 | **63.35** | **61.37** | **72.39** |

The oracle is an upper bound, not a deployable method. It shows that GCR and DVI make different mistakes and that a learned router has meaningful headroom.

## Table 3: Embedding-Guided KG-Trie Results

| Dataset | Method | Setting | Acc./Rec. | Hit | F1 | Precision |
|---|---|---|---:|---:|---:|---:|
| CWQ full | Qwen GCR | 2-hop baseline | 55.60 | 62.50 | **23.86** | 17.73 |
| CWQ full | Qwen embedding | 2-hop baseline-aligned | 55.48 | 62.39 | 23.76 | 17.66 |
| WebQSP full | Qwen GCR | 2-hop baseline | 68.39 | 87.71 | **37.17** | 33.58 |
| WebQSP full | Qwen embedding | 2-hop baseline-aligned | 67.75 | 87.77 | 36.87 | 33.44 |
| CWQ100 | preserve-2 + emb-3 | all candidates | 63.42 | 70.00 | 27.48 | 20.27 |
| CWQ100 | baseline + emb-3 fusion | KGQA top-1 | 39.13 | 47.00 | **39.68** | **45.00** |
| WebQSP100 | preserve-2 + emb-3 | all candidates | 70.89 | 90.00 | 34.18 | 31.02 |
| WebQSP100 | preserve-2 + emb-3 | no-topic top-1 | 42.76 | 66.00 | **44.70** | **64.00** |
| CWQ20 | 3-hop embedding | output top-1 | 43.79 | 55.00 | **45.56** | **55.00** |

Main takeaway: pure 2-hop semantic selection nearly ties but does not beat Qwen GCR. Embedding becomes useful when 2-hop coverage is preserved and 3-hop evidence is added with candidate reranking.

## Table 4: Compositionality-Type Change for Routed DVI

Delta values compare exact-size routed DVI against the corresponding GCR baseline.

| Type | N | Llama-3.1 Delta Hit@1 | Llama-3.1 Delta F1 | Llama-2 Delta Hit@1 | Llama-2 Delta F1 |
|---|---:|---:|---:|---:|---:|
| composition | 1546 | +0.84 | +0.55 | +0.19 | +0.26 |
| conjunction | 1575 | -0.19 | -0.12 | -0.76 | -0.38 |
| comparative | 213 | +1.88 | -0.01 | +0.94 | +0.70 |
| superlative | 197 | -1.02 | -1.92 | +0.00 | +0.54 |

Routed DVI helps some composition and comparative examples, but conjunction and superlative cases remain sensitive to decomposition and answer typing errors.

## Table 5: DVI Gate Threshold Ablation

| Model | Gate | Acc./Rec. | Hit@1 | F1 | Time(s) |
|---|---|---:|---:|---:|---:|
| Llama-3.1 | <code>&#124;C&#124; <= 1</code> | 59.03 | 54.80 | 54.33 | 8.54 |
| Llama-3.1 | <code>&#124;C&#124; = 2</code> | 59.26 | 56.22 | **54.63** | 6.58 |
| Llama-3.1 | <code>&#124;C&#124; <= 2</code> | **59.28** | 55.14 | 54.41 | 10.29 |
| Llama-3.1 | <code>&#124;C&#124; <= 3</code> | 58.92 | 54.63 | 53.87 | 11.27 |
| Llama-2 | <code>&#124;C&#124; <= 1</code> | 60.40 | 55.88 | 54.38 | 6.89 |
| Llama-2 | <code>&#124;C&#124; = 2</code> | **60.56** | **56.92** | **54.51** | 6.53 |
| Llama-2 | <code>&#124;C&#124; <= 2</code> | 60.45 | 55.68 | 54.40 | 7.16 |
| Llama-2 | <code>&#124;C&#124; <= 3</code> | 59.71 | 54.91 | 53.84 | 7.34 |

Exact-size routing, <code>&#124;C&#124;=2</code>, is the most stable deployed gate across the two stronger KG-specialized LLMs.

## Table 6: GraphLite / PathScorer Training Signal

| Verifier | Groups | MRR | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|
| Pretrained cross-encoder | 158 | 68.04 | 56.33 | 85.44 | 88.61 |
| Fine-tuned cross-encoder | 158 | **77.13** | **69.62** | **87.97** | **91.77** |

Fine-tuning improves path-level ranking, but endpoint scoring alone does not guarantee full answer satisfaction.

## Table 7: Hop-Length and Candidate-Selection Ablation

| CWQ20 setting | Hit | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| 2-hop group-beam baseline | 45.00 | 27.83 | 30.00 | 36.88 |
| 2-hop beam early stopping | 45.00 | 30.50 | 34.17 | 36.88 |
| 3-hop budgeted, no embedding | 45.00 | 22.37 | 20.83 | 31.29 |
| 3-hop embedding, all candidates | **60.00** | 37.00 | 37.50 | **51.88** |
| 3-hop embedding, top-1 | 55.00 | **45.56** | **55.00** | 43.79 |

Extra hops hurt when added naively, but help when embedding-guided selection controls path explosion.

## Table 8: Representative Failure Cases

| Case | Required inference steps | Observed KG-valid path / output | Failure point |
|---|---|---|---|
| Country Nation World Tour college | Find the artist of Country Nation World Tour; follow the artist's education relation; return Belmont University. | The model follows a generic type source: College/University -> College -> films -> Johnny Be Good -> country -> United States. | Generic-entity retrieval and answer-type failure. The path is valid but never identifies the tour artist. |
| Afghan National Anthem religions | Map the anthem to its country; from that country, retrieve practiced religions; return Sunni Islam and Shia Islam. | The model follows a topical but wrong relation: Afghan National Anthem -> language -> Pashto language. Some paths continue to Afghanistan but return official language. | Relation-selection failure. The path is about language rather than religion. |
| Libya anthem leader | Identify the country whose anthem is Libya, Libya, Libya; intersect with the office/person constraint from Prime Minister of Libya; return Abdullah al-Thani. | Predicted paths include Prime Minister of Libya -> office holders -> position record -> jurisdiction -> Libya -> languages spoken -> Arabic. | Aggregation and typing failure. The system reaches Libya but drifts to language, not the office holder. |
| Alta Verapaz and Central America | Candidate country must contain Alta Verapaz Department and also be in Central America; return Guatemala. | Baseline follows paths from generic Country, e.g. fictional country settings ending in Europe. | Multi-entity intersection failure. Generic paths overwhelm the specific department/region evidence. |
| Temporal governor filter | Retrieve governors of Arizona; check who held office in 2009; apply the predicate held governmental position before 1998. | Graph paths can retrieve governor/office-holder candidates, but date predicates are not executed by the trie. | Operator failure. Prefix validity does not evaluate temporal constraints. |

These examples show why path validity alone is insufficient: each observed path can be KG-valid but still fail the full question constraint.
