# Code Map

This directory separates the three implementation tracks used in the report.

## `gcr-dvi/`

Main working codebase. It contains the GCR baseline plus the integrated DVI and PathScorer implementations.

Important files:

- `workflow/predict_paths_and_answers.py`: original graph-constrained path generation pipeline.
- `workflow/predict_final_answer.py`: final-answer generation from reasoning paths.
- `workflow/predict_dvi.py`: DVI end-to-end controller.
- `src/dvi/decomposer.py`: question-to-constraint JSON decomposition.
- `src/dvi/intersector.py`: deterministic candidate-set intersection.
- `src/dvi/answer_aware.py`: answer-type refinement and local expansion.
- `src/dvi/path_scorer.py`: GraphLite-style bi-encoder and cross-encoder path scorer.
- `workflow/build_path_verifier_data.py`: creates supervised path-verifier data.
- `workflow/train_path_reranker.py`: trains/evaluates the path reranker.

## `graphlite/`

Standalone entity-level prototype extracted from `lite_framework.rar`. It studies explicit relation selection, entity enumeration, cross-encoder verification, and selective two-hop expansion.

Important files:

- `lite_framework/entity_level_decoder.py`: main graph enumeration and entity scoring logic.
- `lite_framework/graph_walker.py`: local graph traversal.
- `lite_framework/graph_encoder.py`: graph/path encoding helpers.
- `lite_framework/cross_encoder_verifier.py`: DeBERTa verifier wrapper.
- `lite_framework/train_cross_encoder.py`: relation verifier training.
- `lite_framework/train_entity_verifier.py`: entity verifier training.

## `embedding_kgtrie/`

Overlay patch for embedding-guided KG-Trie construction. It changes how candidate paths are selected before constrained decoding.

Important files:

- `kgtrie_enhanced_changes.patch`: full patch against the base GCR-style tree.
- `overlay/src/qa_prompt_builder.py`: semantic path ranking and trie construction changes.
- `overlay/workflow/predict_paths_and_answers.py`: workflow entry point with enhanced path selection arguments.

The full raw archive was not copied here because it contains multi-GB experiment outputs. See the final report and `../results_summary/RESULTS.md` for the compact record of the experiments.
