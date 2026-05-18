# GraphLite Prototype

This directory contains the GraphLite entity-level prototype extracted from `lite_framework.rar`.

GraphLite replaces token-by-token KG-LLM path generation with a structured verifier:

1. Enumerate candidate relations from topic entities.
2. Score question-relation pairs with a cross-encoder.
3. Enumerate candidate entities from selected relations.
4. Score question-path-entity triples with an entity verifier.
5. Optionally expand selected candidates to two hops.

Core files are under `lite_framework/`. The integrated, lighter-weight PathScorer variant used by the DVI codebase is in `../dvi_gcr/src/dvi/path_scorer.py`.


Quick CLI checks from `code/graphlite/lite_framework/`:

```bash
python entity_level_predict.py --help
python train_cross_encoder.py --help
python train_entity_verifier.py --help
```

The standalone scripts default to `--data_path rmanluo`, matching `datasets/DATASET_LINKS.md`. Full runs require model checkpoints or trained verifier checkpoints.
