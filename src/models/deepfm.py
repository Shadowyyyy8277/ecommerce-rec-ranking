"""PyTorch DeepFM model implementation."""

from __future__ import annotations

import torch
from torch import nn


class DeepFM(nn.Module):
    """DeepFM model with sparse embeddings, FM interactions, and dense features."""

    def __init__(self, field_dims: list[int], num_dense_features: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.field_dims = field_dims
        self.num_dense_features = num_dense_features
        self.embedding_dim = embedding_dim

        self.linear_embeddings = nn.ModuleList([nn.Embedding(dim, 1) for dim in field_dims])
        self.fm_embeddings = nn.ModuleList([nn.Embedding(dim, embedding_dim) for dim in field_dims])
        self.dense_linear = nn.Linear(num_dense_features, 1) if num_dense_features > 0 else None

        dnn_input_dim = len(field_dims) * embedding_dim + num_dense_features
        self.dnn = nn.Sequential(
            nn.Linear(dnn_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def forward(self, sparse_x: torch.Tensor, dense_x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of sparse and dense features."""
        linear_terms = [emb(sparse_x[:, i]) for i, emb in enumerate(self.linear_embeddings)]
        linear_out = torch.stack(linear_terms, dim=0).sum(dim=0)
        if self.dense_linear is not None:
            linear_out = linear_out + self.dense_linear(dense_x)

        embed_list = [emb(sparse_x[:, i]) for i, emb in enumerate(self.fm_embeddings)]
        fm_stack = torch.stack(embed_list, dim=1)
        square_of_sum = torch.sum(fm_stack, dim=1) ** 2
        sum_of_square = torch.sum(fm_stack ** 2, dim=1)
        fm_out = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        dnn_input = torch.cat([fm_stack.flatten(start_dim=1), dense_x], dim=1)
        dnn_out = self.dnn(dnn_input)
        logits = linear_out + fm_out + dnn_out + self.bias
        return logits.squeeze(1)

    def _init_weights(self) -> None:
        """Initialize embeddings and linear layers."""
        for emb in list(self.linear_embeddings) + list(self.fm_embeddings):
            nn.init.xavier_uniform_(emb.weight.data)
        for module in self.dnn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                nn.init.zeros_(module.bias.data)
        if self.dense_linear is not None:
            nn.init.xavier_uniform_(self.dense_linear.weight.data)
            nn.init.zeros_(self.dense_linear.bias.data)
