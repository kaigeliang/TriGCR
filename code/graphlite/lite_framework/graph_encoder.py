"""
Encoder-Scored Graph Search: score KG edges using LLM's full encoder (32 layers),
not static token embeddings. One batched forward pass scores all edges at once.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import networkx as nx


class EncoderGraphScorer:
    """Score KG edges by running the LLM encoder on batched edge descriptions."""

    def __init__(self, model, tokenizer):
        self.model = model  # HuggingFace LlamaForCausalLM
        self.tok = tokenizer
        self.device = next(model.parameters()).device

    @staticmethod
    def _rel_to_nl(rel: str) -> str:
        return rel.rsplit('.', 1)[-1].replace('_s', '').replace('_', ' ')

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of texts → mean-pooled last hidden states [N, D]."""
        # Tokenize with padding
        enc = self.tok(texts, return_tensors='pt', padding=True, truncation=True,
                       max_length=64, add_special_tokens=False)
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)

        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # Last hidden layer [N, max_len, D]
        hidden = outputs.last_hidden_state

        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()  # [N, max_len, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [N, D]
        return F.normalize(pooled, p=2, dim=1)

    def score_edges(
        self,
        graph: nx.DiGraph,
        topic_entities: list[str],
        question: str,
    ) -> dict[tuple[str, str, str], float]:
        """
        Score all outgoing edges from nodes reachable from topic entities.

        Returns:
            dict mapping (head, relation, tail) → score in [0, 1].
        """
        # Collect all reachable edges (1-hop and 2-hop)
        edges: set[tuple[str, str, str]] = set()
        reachable_nodes = set(topic_entities)
        first_hop_entities = set()

        for e in topic_entities:
            if e not in graph:
                continue
            for nb in graph.neighbors(e):
                rel = graph[e][nb]["relation"]
                edges.add((e, rel, nb))
                first_hop_entities.add(nb)

        for e in first_hop_entities:
            if e not in graph:
                continue
            for nb in graph.neighbors(e):
                rel = graph[e][nb]["relation"]
                edges.add((e, rel, nb))
                reachable_nodes.add(nb)

        if not edges:
            return {}

        edge_list = list(edges)

        # Build edge descriptions
        q_prefix = question.strip().rstrip('?')
        edge_texts = [
            f"Question: {q_prefix}. Edge: {self._rel_to_nl(r)} leads to {t}."
            for h, r, t in edge_list
        ]

        # Encode question alone (as reference)
        q_text = f"Question: {q_prefix}"
        q_vec = self._encode_texts([q_text])  # [1, D]

        # Encode all edges in one batch
        batch_size = 256
        all_sims = []
        for i in range(0, len(edge_texts), batch_size):
            batch = edge_texts[i:i + batch_size]
            e_vecs = self._encode_texts(batch)  # [B, D]
            sims = (e_vecs @ q_vec.T).squeeze(1)  # [B]
            all_sims.append(sims)

        all_sims = torch.cat(all_sims)
        # Normalize scores to [0, 1]
        min_s, max_s = all_sims.min(), all_sims.max()
        if max_s > min_s:
            all_sims = (all_sims - min_s) / (max_s - min_s)

        return {edge_list[i]: float(all_sims[i]) for i in range(len(edge_list))}
