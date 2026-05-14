# KG-Trie Enhanced Modifications

This package keeps all project modifications outside the upstream `graph-constrained-reasoning` source tree.

## Apply

From a clean `graph-constrained-reasoning` checkout:

```bash
git apply /mnt/data/kaigeliang/kgtrie/custom_code/kgtrie_enhanced_mods/kgtrie_enhanced_changes.patch
```

Or copy files from `overlay/` into matching repository-relative paths.

## Added Methods

- `--embedding_guided`: question-path embedding reranking before KG-Trie construction.
- `--hybrid_guided`: embedding + relation keyword + entity keyword + shared-tail coverage + source-specificity path ranking.
- `--filter_generic_sources`: removes paths from generic type-like topic entities when more specific topic entities exist.
- `--output_topk`: keeps only top-k generated candidates before saving/evaluation; `--output_topk 1` turns the top-1 analysis into an actual precision-oriented system output.
- `dfs_limited`: bounded path collection for high-hop experiments.

## Best CWQ20 Result

Embedding-guided 3-hop KG-Trie with `--output_topk 1`:

- Hit: 55.00
- Answer F1: 45.56
- Precision: 55.00
- Recall: 43.79

Output directory:

`/mnt/data/kaigeliang/kgtrie/experiment_results/kgtrie_enhanced_worktree/results/GenPaths/RoG-cwq/GCR-Qwen2-0.5B-Instruct/test[:20]/emb-top1-zero-shot-beam-early-stopping-k3-index_len3/`
