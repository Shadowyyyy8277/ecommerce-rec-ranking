"""ItemCF recall based on normalized item co-occurrence."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def build_itemcf_similarity(train_df: pd.DataFrame, max_seq_len: int = 80, top_k: int = 80) -> Dict[str, List[Tuple[str, float]]]:
    """Build item-to-item similarity using user behavior sequences."""
    df = train_df.sort_values("timestamp")
    sequences = df.groupby("user_id")["item_id"].apply(list)
    co_counts: Dict[str, Counter] = defaultdict(Counter)
    item_counts: Counter = Counter()

    for seq in sequences:
        recent_seq = _dedupe_keep_order(seq[-max_seq_len:])
        for item_i in recent_seq:
            item_counts[item_i] += 1
        for idx_i, item_i in enumerate(recent_seq):
            for idx_j, item_j in enumerate(recent_seq):
                if item_i == item_j:
                    continue
                distance = abs(idx_i - idx_j)
                co_counts[item_i][item_j] += 1.0 / (1.0 + distance)

    similarity: Dict[str, List[Tuple[str, float]]] = {}
    for item_i, related in co_counts.items():
        sims = []
        for item_j, cij in related.items():
            score = cij / sqrt(float(item_counts[item_i]) * float(item_counts[item_j]) + 1e-9)
            sims.append((item_j, float(score)))
        similarity[item_i] = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
    return similarity


def build_itemcf_recall(train_df: pd.DataFrame, users: Iterable[str] | None = None, top_k: int = 50) -> pd.DataFrame:
    """Recall items similar to each user's historical items."""
    similarity = build_itemcf_similarity(train_df, top_k=max(top_k * 2, 50))
    df = train_df.sort_values("timestamp")
    user_histories = df.groupby("user_id")["item_id"].apply(list).to_dict()
    user_list = list(users) if users is not None else sorted(user_histories)

    rows = []
    for user_id in user_list:
        history_seq = user_histories.get(user_id, [])
        history_set = set(history_seq)
        scores: Counter = Counter()
        for weight_rank, item_id in enumerate(reversed(history_seq[-30:]), start=1):
            recency_weight = 1.0 / sqrt(weight_rank)
            for cand_item, sim_score in similarity.get(item_id, []):
                if cand_item not in history_set:
                    scores[cand_item] += sim_score * recency_weight
        for item_id, score in scores.most_common(top_k):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "recall_score": float(score),
                    "recall_type": "itemcf",
                }
            )
    return pd.DataFrame(rows)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """De-duplicate a sequence while keeping the last observed order."""
    seen = set()
    output = []
    for item in reversed(items):
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return list(reversed(output))
