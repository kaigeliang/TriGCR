"""
Entity-Level Decoding v2: batch score path prefixes by LLM perplexity.
Replaces MC-selection with the model's own path-prediction confidence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import networkx as nx


class EntityLevelDecoder:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        if model is not None:
            self.device = next(model.parameters()).device
        else:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verifier = None
        self.entity_verifier = None

    def load_verifier(self, verifier_path: str, model_name: str = 'microsoft/deberta-v3-base'):
        """Load a trained cross-encoder verifier for candidate reranking."""
        from cross_encoder_verifier import CrossEncoderVerifier
        self.verifier = CrossEncoderVerifier.load(verifier_path, model_name, self.device)

    def load_entity_verifier(self, verifier_path: str, model_name: str = 'microsoft/deberta-v3-base'):
        """Load a trained entity-level verifier for entity scoring."""
        from cross_encoder_verifier import CrossEncoderVerifier
        self.entity_verifier = CrossEncoderVerifier.load(verifier_path, model_name, self.device)

    @staticmethod
    def _rel_to_nl(rel: str) -> str:
        return rel.rsplit('.', 1)[-1].replace('_s', '').replace('_', ' ')

    @torch.no_grad()
    def _pairwise_compare(self, question: str, candidates: list[str],
                          max_batch: int = 24) -> list[float]:
        """Score N candidates by pairwise tournament. Returns win count per candidate."""
        import random
        n = len(candidates)
        if n <= 1:
            return [0.0] * n

        # Shuffle and pair up
        indices = list(range(n))
        random.shuffle(indices)
        pairs = [(indices[i], indices[i+1]) for i in range(0, n-1, 2)]

        # Build comparison prompts
        prompts = []
        pair_map = []  # (i, j) for each prompt
        q_short = question.strip().rstrip('?')
        for i, j in pairs:
            prompts.append(
                f'Question: {q_short}?\n'
                f'A: {candidates[i]}\n'
                f'B: {candidates[j]}\n'
                f'Which is more relevant? A or B?\nAnswer:')
            pair_map.append((i, j))

        # Score in batches
        a_id = self.tok.convert_tokens_to_ids('A')
        b_id = self.tok.convert_tokens_to_ids('B')
        wins = [0.0] * n

        for start in range(0, len(prompts), max_batch):
            batch = prompts[start:start + max_batch]
            enc = self.tok(batch, return_tensors='pt', padding=True, add_special_tokens=False)
            input_ids = enc.input_ids.to(self.device)
            logits = self.model(input_ids).logits  # [B, L, V]
            last_logits = logits[:, -1, :]  # [B, V]

            for k, (i, j) in enumerate(pair_map[start:start + max_batch]):
                score_a = float(last_logits[k, a_id]) if a_id < last_logits.shape[1] else 0.0
                score_b = float(last_logits[k, b_id]) if b_id < last_logits.shape[1] else 0.0
                if score_a > score_b:
                    wins[i] += 1.0
                elif score_b > score_a:
                    wins[j] += 1.0
                # tie: both get 0

        return wins

    @torch.no_grad()
    def _score_hidden_sim(self, prefixes: list[str], skip_tokens: int = 0,
                          max_batch: int = 16) -> list[float]:
        """Score by cosine sim between question hidden states and path hidden states."""
        all_scores = []
        for start in range(0, len(prefixes), max_batch):
            batch = prefixes[start:start + max_batch]
            enc = self.tok(batch, return_tensors='pt', padding=True, add_special_tokens=False)
            input_ids = enc.input_ids.to(self.device)
            mask = enc.attention_mask.to(self.device)
            outputs = self.model.model(
                input_ids=input_ids, attention_mask=mask,
                use_cache=False)
            hidden = outputs.last_hidden_state  # [B, L, D]

            for b in range(input_ids.shape[0]):
                seq_len = mask[b].sum().item()
                if skip_tokens >= seq_len or seq_len - skip_tokens < 2:
                    all_scores.append(0.0)
                    continue
                # Question part: up to skip_tokens. Path part: after skip_tokens.
                q_vec = hidden[b, :skip_tokens].mean(dim=0)  # [D]
                p_vec = hidden[b, skip_tokens:seq_len].mean(dim=0)  # [D]
                sim = float(F.cosine_similarity(q_vec.unsqueeze(0), p_vec.unsqueeze(0)))
                all_scores.append(sim)
        return all_scores

    @torch.no_grad()
    def _score_prefixes(self, prefixes: list[str], skip_tokens: int = 0,
                        custom_skips: list[int] | None = None,
                        max_batch: int = 16) -> list[float]:
        """Score tokens after skip point. custom_skips overrides skip_tokens per-item."""
        all_scores = []
        for start in range(0, len(prefixes), max_batch):
            batch = prefixes[start:start + max_batch]
            enc = self.tok(batch, return_tensors='pt', padding=True, add_special_tokens=False)
            input_ids = enc.input_ids.to(self.device)
            mask = enc.attention_mask.to(self.device)
            log_probs = F.log_softmax(self.model(input_ids).logits, dim=-1)

            for b in range(input_ids.shape[0]):
                seq_len = mask[b].sum().item()
                item_skip = custom_skips[start + b] if custom_skips else skip_tokens
                start_pos = min(item_skip, seq_len - 2)
                if start_pos >= seq_len - 1:
                    all_scores.append(-float('inf'))
                    continue
                total = 0.0
                for i in range(start_pos, seq_len - 1):
                    tok = input_ids[b, i + 1]
                    total += log_probs[b, i, tok].item()
                all_scores.append(total)
        return all_scores

    def decode(
        self,
        graph: nx.DiGraph,
        topic_entities: list[str],
        question: str,
        max_depth: int = 2,
        top_k: int = 5,
    ) -> list[tuple[str, list[tuple[str, str, str]], float]]:
        results: dict[str, tuple[list, float]] = {}

        # --- Score unique relations first (not all edges) ---
        q_prefix = question.strip().rstrip('?')
        prompt_base = (f"# Question:\n{q_prefix}?\n"
                       f"# Topic entities:\n{', '.join(topic_entities)}\n")

        # Collect unique relations and their edges
        rel_to_edges: dict[str, list[tuple[str, str, str]]] = {}
        for e in topic_entities:
            if e not in graph:
                continue
            for nb in graph.neighbors(e):
                rel = graph[e][nb]["relation"]
                nl = self._rel_to_nl(rel)
                if nl not in rel_to_edges:
                    rel_to_edges[nl] = []
                rel_to_edges[nl].append((e, rel, nb))

        if not rel_to_edges:
            return []

        # Pre-compute prompt token count to skip in scoring
        prompt_tok_len = len(self.tok.encode(prompt_base, add_special_tokens=False))

        # Score relations: use verifier if loaded, else hidden-state similarity
        rel_names = list(rel_to_edges.keys())
        if self.verifier is not None:
            rel_scores = self.verifier.score([question] * len(rel_names), rel_names)
        else:
            rel_prefixes = [prompt_base + f"# Relation: {nl}" for nl in rel_names]
            rel_scores = self._score_hidden_sim(rel_prefixes, skip_tokens=prompt_tok_len)
        ranked_rels = sorted(zip(rel_names, rel_scores), key=lambda x: -x[1])

        # For top-K relations, score entities after # Answer:\n (GCR-aligned)
        entity_prefixes: list[str] = []
        entity_edges: list[tuple[str, str, str, str]] = []  # (nl, h, rel, t)
        entity_skips: list[int] = []
        for nl, _ in ranked_rels[:top_k]:
            for h, rel, t in rel_to_edges[nl]:
                if self.entity_verifier is not None:
                    # Entity verifier: entity in path text (matches training format)
                    prefix = (prompt_base +
                              f"# Reasoning Path:\n{h} -> {nl} -> {t}\n# Answer:\n{t}")
                    skip = len(self.tok.encode(
                        prompt_base + f"# Reasoning Path:\n{h} -> {nl} -> {t}\n# Answer:\n",
                        add_special_tokens=False))
                else:
                    # Llama: entity ONLY after # Answer: (GCR-aligned, no duplication)
                    prefix = (prompt_base +
                              f"# Reasoning Path:\n{h} -> {nl}\n# Answer:\n{t}")
                    skip = len(self.tok.encode(
                        prompt_base + f"# Reasoning Path:\n{h} -> {nl}\n# Answer:\n",
                        add_special_tokens=False))
                entity_prefixes.append(prefix)
                entity_edges.append((nl, h, rel, t))
                entity_skips.append(skip)

        if entity_prefixes:
            if self.entity_verifier is not None:
                path_strs = []
                for nl, h, rel, t in entity_edges:
                    path_strs.append(f'{h} -> {rel} -> {t}')
                entity_scores = self.entity_verifier.score(
                    [question] * len(path_strs), path_strs)
                if not hasattr(self, '_ev_debug_done'):
                    self._ev_debug_done = True
                    print(f'[EV debug] {len(path_strs)} entities, scores: '
                          f'min={min(entity_scores):.3f} max={max(entity_scores):.3f} '
                          f'mean={sum(entity_scores)/len(entity_scores):.3f}')
            else:
                entity_scores = []
                for i in range(0, len(entity_prefixes), 16):
                    batch = entity_prefixes[i:i+16]
                    skips = entity_skips[i:i+16]
                    entity_scores.extend(
                        self._score_prefixes(batch, skip_tokens=None, custom_skips=skips))
        else:
            entity_scores = []

        # 2-hop: expand from ALL entities (not just MIDs)
        prefixes_2hop: list[str] = []
        edges_2hop: list[tuple[str, str, str, str, str, str]] = []
        skips_2hop: list[int] = []
        for (nl, h, rel, t), e_score in zip(entity_edges, entity_scores):
            if t not in results or e_score > results[t][1]:
                results[t] = ([(h, rel, t)], e_score)
            is_mid = t.startswith('m.') or t.startswith('g.')
            if self.entity_verifier is not None:
                expand = is_mid or e_score > 0.3
            else:
                expand = is_mid or e_score > -20.0  # Llama log P: filter garbage MIDs
            if t in graph and expand:
                for nb in graph.neighbors(t):
                    rel2 = graph[t][nb]["relation"]
                    nl2 = self._rel_to_nl(rel2)
                    # Entity after # Answer:\n (GCR-aligned). For Llama: omit final entity from path.
                    if self.entity_verifier is not None:
                        prefix_2h = (prompt_base +
                            f"# Reasoning Path:\n{h} -> {nl} -> {t} -> {nl2} -> {nb}\n# Answer:\n{nb}")
                        sk = len(self.tok.encode(
                            prompt_base + f"# Reasoning Path:\n{h} -> {nl} -> {t} -> {nl2} -> {nb}\n# Answer:\n",
                            add_special_tokens=False))
                    else:
                        prefix_2h = (prompt_base +
                            f"# Reasoning Path:\n{h} -> {nl} -> {t} -> {nl2}\n# Answer:\n{nb}")
                        sk = len(self.tok.encode(
                            prompt_base + f"# Reasoning Path:\n{h} -> {nl} -> {t} -> {nl2}\n# Answer:\n",
                            add_special_tokens=False))
                    prefixes_2hop.append(prefix_2h)
                    edges_2hop.append((nl, h, rel, t, rel2, nl2, nb))
                    skips_2hop.append(sk)

        if prefixes_2hop:
            if self.entity_verifier is not None:
                path_strs_2h = []
                for nl, h, rel, t, rel2, nl2, nb in edges_2hop:
                    path_strs_2h.append(f'{h} -> {rel} -> {t} -> {rel2} -> {nb}')
                scores_2hop = self.entity_verifier.score(
                    [question] * len(path_strs_2h), path_strs_2h)
            else:
                scores_2hop = []
                for i in range(0, len(prefixes_2hop), 16):
                    batch = prefixes_2hop[i:i+16]
                    skips = skips_2hop[i:i+16]
                    scores_2hop.extend(
                        self._score_prefixes(batch, skip_tokens=None, custom_skips=skips))
            for (nl, h, rel, t, rel2, nl2, nb), s2 in zip(edges_2hop, scores_2hop):
                if nb not in results or s2 > results[nb][1]:
                    results[nb] = ([(h, rel, t), (t, rel2, nb)], s2)

        # Cross-encoder reranking (only if entity_verifier is not loaded)
        if self.verifier is not None and self.entity_verifier is None and results:
            entities = list(results.keys())
            path_strs = []
            for e in entities:
                triples = results[e][0]
                path_strs.append(' -> '.join(
                    f'{h} -> {r} -> {t}'
                    for h, r, t in triples))
            v_scores = self.verifier.score(
                [question] * len(entities), path_strs)
            for e, vs in zip(entities, v_scores):
                prev_path, _ = results[e]
                results[e] = (prev_path, float(vs))

        sorted_results = sorted(results.items(), key=lambda x: -x[1][1])
        return [(e, p, s) for e, (p, s) in sorted_results[:top_k * 3]]
