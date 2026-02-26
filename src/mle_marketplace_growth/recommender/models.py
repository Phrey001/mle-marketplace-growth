from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


def _popularity_scores(train: dict[str, set[str]], item_to_idx: dict[str, int]) -> np.ndarray:
    scores = np.zeros(len(item_to_idx), dtype=float)
    for items in train.values():
        for item_id in items:
            scores[item_to_idx[item_id]] += 1.0
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def _train_mf(train: dict[str, set[str]], user_to_idx: dict[str, int], item_to_idx: dict[str, int], n_components: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(user_to_idx), len(item_to_idx)), dtype=float)
    for user_id, items in train.items():
        user_idx = user_to_idx[user_id]
        for item_id in items:
            matrix[user_idx, item_to_idx[item_id]] = 1.0
    effective_components = max(2, min(n_components, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=effective_components, random_state=42)
    return svd.fit_transform(matrix), svd.components_.T


def _train_two_tower(
    train: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    embedding_dim: int,
    epochs: int,
    learning_rate: float,
    negative_samples: int,
    l2_reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(42)
    torch.manual_seed(42)

    positives: list[tuple[int, int]] = []
    seen_item_indices: dict[int, set[int]] = {}
    for user_id, items in train.items():
        user_idx = user_to_idx[user_id]
        seen = {item_to_idx[item_id] for item_id in items if item_id in item_to_idx}
        seen_item_indices[user_idx] = seen
        for item_idx in seen:
            positives.append((user_idx, item_idx))

    if not positives: raise ValueError("No positive interactions available for two-tower training.")

    user_embedding_layer = torch.nn.Embedding(len(user_to_idx), embedding_dim)
    item_embedding_layer = torch.nn.Embedding(len(item_to_idx), embedding_dim)
    with torch.no_grad():
        user_embedding_layer.weight.normal_(mean=0.0, std=0.05)
        item_embedding_layer.weight.normal_(mean=0.0, std=0.05)
    optimizer = torch.optim.SGD(list(user_embedding_layer.parameters()) + list(item_embedding_layer.parameters()), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        rng.shuffle(positives)
        for user_idx, pos_item_idx in positives:
            seen = seen_item_indices[user_idx]
            neg_item_indices = []
            for _ in range(negative_samples):
                candidate = rng.randrange(len(item_to_idx))
                while candidate in seen:
                    candidate = rng.randrange(len(item_to_idx))
                neg_item_indices.append(candidate)

            batch_user = torch.tensor([user_idx] * (1 + negative_samples), dtype=torch.long)
            batch_item = torch.tensor([pos_item_idx] + neg_item_indices, dtype=torch.long)
            batch_label = torch.tensor([1.0] + [0.0] * negative_samples, dtype=torch.float32)
            user_vec, item_vec = user_embedding_layer(batch_user), item_embedding_layer(batch_item)
            loss = loss_fn((user_vec * item_vec).sum(dim=1), batch_label)
            if l2_reg > 0: loss = loss + l2_reg * (user_vec.pow(2).mean() + item_vec.pow(2).mean())
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    return user_embedding_layer.weight.detach().cpu().numpy(), item_embedding_layer.weight.detach().cpu().numpy()
