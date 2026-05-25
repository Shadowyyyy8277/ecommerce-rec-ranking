"""Lightweight Item2Vec recall using PyTorch skip-gram with negative sampling."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_item2vec_recall(
    train_df: pd.DataFrame,
    users: Iterable[str] | None = None,
    top_k: int = 50,
    embedding_dim: int = 32,
    epochs: int = 2,
    seed: int = 42,
    batch_size: int = 8192,
    window_size: int = 2,
    max_pairs: int = 8_000,
) -> pd.DataFrame:
    """Train lightweight item embeddings and recall nearest items per user."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(seed)
    df = train_df.sort_values("timestamp")
    item_ids = sorted(df["item_id"].unique())
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    sequences = df.groupby("user_id")["item_id"].apply(list).to_dict()

    pairs = _build_skipgram_pairs(sequences, item_to_idx, window_size=window_size, max_pairs=max_pairs, rng=rng)
    if not pairs:
        return pd.DataFrame(columns=["user_id", "item_id", "recall_score", "recall_type"])

    count_map = df["item_id"].value_counts().to_dict()
    item_counts = np.array([max(1, int(count_map.get(item_id, 1))) for item_id in item_ids], dtype=np.float64)
    neg_probs = np.power(item_counts, 0.75)
    neg_probs = neg_probs / neg_probs.sum()

    centers = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    dataset = TensorDataset(centers, contexts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _SkipGramNegSampling(num_items=len(item_ids), embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    neg_probs_tensor = torch.tensor(neg_probs, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    model.train()
    for _ in range(epochs):
        progress = tqdm(loader, desc="item2vec", leave=False)
        for center_batch, context_batch in progress:
            center_batch = center_batch.to(device)
            context_batch = context_batch.to(device)
            neg_batch = torch.multinomial(neg_probs_tensor, center_batch.shape[0] * 2, replacement=True).view(center_batch.shape[0], 2)
            loss = model(center_batch, context_batch, neg_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}"})

    embeddings = model.input_embeddings.weight.detach().cpu().numpy()
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    user_list = list(users) if users is not None else sorted(sequences)
    rows = []
    for user_id in user_list:
        history = sequences.get(user_id, [])
        history_indices = [item_to_idx[item] for item in history[-30:] if item in item_to_idx]
        if not history_indices:
            continue
        user_vec = embeddings[history_indices].mean(axis=0)
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-9)
        scores = embeddings @ user_vec
        history_set = set(history_indices)
        candidate_count = min(len(item_ids), top_k + len(history_set) + 50)
        top_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        ranked = sorted([(idx, float(scores[idx])) for idx in top_indices if idx not in history_set], key=lambda x: x[1], reverse=True)[:top_k]
        for idx, score in ranked:
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": idx_to_item[idx],
                    "recall_score": score,
                    "recall_type": "item2vec",
                }
            )
    return pd.DataFrame(rows)


class _SkipGramNegSampling:
    """Tiny wrapper class assigned dynamically when torch is available."""

    def __new__(cls, num_items: int, embedding_dim: int):
        import torch
        from torch import nn

        class _TorchSkipGram(nn.Module):
            """Skip-gram model with negative sampling loss."""

            def __init__(self, n_items: int, dim: int) -> None:
                super().__init__()
                self.input_embeddings = nn.Embedding(n_items, dim)
                self.output_embeddings = nn.Embedding(n_items, dim)
                nn.init.xavier_uniform_(self.input_embeddings.weight)
                nn.init.xavier_uniform_(self.output_embeddings.weight)

            def forward(self, center, context, negatives):
                center_emb = self.input_embeddings(center)
                context_emb = self.output_embeddings(context)
                positive_score = torch.sum(center_emb * context_emb, dim=1)
                positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-9)
                negative_emb = self.output_embeddings(negatives)
                negative_score = torch.bmm(negative_emb.neg(), center_emb.unsqueeze(2)).squeeze(2)
                negative_loss = -torch.log(torch.sigmoid(negative_score) + 1e-9).sum(dim=1)
                return (positive_loss + negative_loss).mean()

        return _TorchSkipGram(num_items, embedding_dim)


def _build_skipgram_pairs(
    sequences: Dict[str, List[str]],
    item_to_idx: Dict[str, int],
    window_size: int,
    max_pairs: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Build skip-gram center-context pairs from user item sequences."""
    pairs: List[Tuple[int, int]] = []
    for seq in sequences.values():
        indexed = [item_to_idx[item] for item in seq[-80:] if item in item_to_idx]
        for i, center in enumerate(indexed):
            left = max(0, i - window_size)
            right = min(len(indexed), i + window_size + 1)
            for j in range(left, right):
                if i != j:
                    pairs.append((center, indexed[j]))
    if len(pairs) > max_pairs:
        sample_indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in sample_indices]
    return pairs
