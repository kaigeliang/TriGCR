"""
Cross-Encoder Path Verifier: scores (question, path) pairs for KGQA relevance.

Lightweight cross-encoder (DeBERTa-v3 or similar) with a classification head,
trained to distinguish correct KG paths from wrong ones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CrossEncoderVerifier(nn.Module):
    """Small transformer with binary classification head for path verification."""

    def __init__(self, model_name: str = 'microsoft/deberta-v3-base'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
        )

    def _format_input(self, question: str, path: str) -> str:
        return f"Question: {question}\nPath: {path}\nRelevant?"

    @torch.no_grad()
    def score(self, questions: list[str], paths: list[str],
              max_batch: int = 32) -> list[float]:
        """Score a batch of (question, path) pairs. Returns [0, 1] scores."""
        self.eval()
        all_scores = []
        for start in range(0, len(questions), max_batch):
            batch_q = questions[start:start + max_batch]
            batch_p = paths[start:start + max_batch]
            texts = [self._format_input(q, p) for q, p in zip(batch_q, batch_p)]
            enc = self.tokenizer(texts, return_tensors='pt', padding=True,
                                 truncation=True, max_length=256)
            enc = {k: v.to(next(self.parameters()).device) for k, v in enc.items()}
            outputs = self.encoder(**enc)
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            logits = self.classifier(pooled)  # [B, 1]
            scores = torch.sigmoid(logits).squeeze(-1)
            all_scores.extend(scores.tolist())
        return all_scores

    def forward(self, questions: list[str], paths: list[str]):
        """Training forward pass. Returns logits for BCE loss."""
        texts = [self._format_input(q, p) for q, p in zip(questions, paths)]
        enc = self.tokenizer(texts, return_tensors='pt', padding=True,
                             truncation=True, max_length=256)
        enc = {k: v.to(next(self.parameters()).device) for k, v in enc.items()}
        outputs = self.encoder(**enc)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled).squeeze(-1)

    def save(self, path: str):
        torch.save({'model': self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, model_name: str = 'microsoft/deberta-v3-base',
             device: str = 'cuda') -> 'CrossEncoderVerifier':
        model = cls(model_name)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()
        return model
