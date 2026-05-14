"""
Dual-Encoder Graph Traversal: search KG by embedding similarity, not text generation.

Encodes questions, relations, and entities using the LLM's own token embedding
table (4096-dim, fine-tuned for KG reasoning).  Walks the graph greedily using
cosine similarity as a search heuristic — no beam-search text generation needed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import networkx as nx


class GraphWalker:
    def __init__(self, embed_weight: torch.Tensor, tokenizer):
        """
        Args:
            embed_weight: LLM token embedding matrix [vocab, dim] on CPU.
            tokenizer: HuggingFace tokenizer.
        """
        self.emb = embed_weight  # [V, D] on CPU
        self.tok = tokenizer
        self.dim = embed_weight.shape[1]
        self._cache: dict[str, torch.Tensor] = {}

    def _encode(self, text: str) -> torch.Tensor:
        """Mean-pool token embeddings for a short text (with cache)."""
        if text not in self._cache:
            ids = self.tok.encode(text, add_special_tokens=False)
            if not ids:
                self._cache[text] = torch.zeros(self.dim)
            else:
                self._cache[text] = self.emb[ids].mean(dim=0)
        return self._cache[text]

    def _score_edge(self, q_emb: torch.Tensor, rel: str, target_entity: str) -> float:
        """Score edge by cosine sim of question to relation (NL form, primary) and entity (bonus)."""
        nl_rel = rel.rsplit('.', 1)[-1].replace('_s', '').replace('_', ' ')
        r_emb = self._encode(nl_rel)
        r_sim = float(F.cosine_similarity(q_emb.unsqueeze(0), r_emb.unsqueeze(0)))
        e_emb = self._encode(target_entity)
        e_sim = float(F.cosine_similarity(q_emb.unsqueeze(0), e_emb.unsqueeze(0)))
        return 0.7 * r_sim + 0.3 * e_sim

    def walk(
        self,
        graph: nx.DiGraph,
        topic_entities: list[str],
        question: str,
        max_depth: int = 2,
        beam_width: int = 10,
        max_candidates: int = 50,
    ) -> list[tuple[str, list[tuple[str, str, str]], float]]:
        """
        Walk the graph from topic entities, guided by embedding similarity.

        Returns:
            List of (answer_entity, path_triples, score) sorted by score desc.
        """
        q_emb = self._encode(question)
        # Normalize once
        q_emb = F.normalize(q_emb, p=2, dim=0)

        # Path queue: (path_triples, cumulative_score)
        # path_triples: [(h, r, t), ...]
        paths = []

        # Initialize from topic entities
        for e in topic_entities:
            if e not in graph:
                continue
            paths.append(([], e, 1.0))  # (triples_so_far, current_node, score)

        for depth in range(max_depth):
            if not paths:
                break

            # Score all outgoing edges from all current paths
            candidates = []
            for triples, node, path_score in paths:
                for nb in graph.neighbors(node):
                    rel = graph[node][nb]["relation"]
                    edge_score = self._score_edge(q_emb, rel, nb)
                    new_score = path_score * (0.3 + 0.7 * edge_score)
                    new_triples = triples + [(node, rel, nb)]
                    candidates.append((new_triples, nb, new_score))

            if not candidates:
                break

            # Keep top-K by score
            candidates.sort(key=lambda x: -x[2])
            paths = candidates[:beam_width]

        # Collect terminal entities with best paths
        results: dict[str, tuple[list, float]] = {}  # entity -> (best_path, best_score)
        for triples, node, score in paths:
            if triples:  # non-empty path
                if node not in results or score > results[node][1]:
                    results[node] = (triples, score)

        # Sort and return top candidates
        sorted_results = sorted(results.items(), key=lambda x: -x[1][1])
        return [(entity, path, score) for entity, (path, score) in sorted_results[:max_candidates]]
