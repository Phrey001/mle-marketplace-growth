from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.decomposition import TruncatedSVD


def _popularity_scores(train: dict[str, set[str]], item_to_idx: dict[str, int], transform: str = "linear") -> np.ndarray:
    scores = np.zeros(len(item_to_idx), dtype=float)
    for items in train.values():
        for item_id in items:
            scores[item_to_idx[item_id]] += 1.0
    if transform == "log1p":
        scores = np.log1p(scores)
    elif transform != "linear":
        raise ValueError(f"Unsupported popularity transform: {transform}")
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def _train_mf(
    train: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    n_components: int,
    n_iter: int = 15,
    weighting: str = "tfidf",
    algorithm: str = "randomized",
    tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(user_to_idx), len(item_to_idx)), dtype=float)
    for user_id, items in train.items():
        user_idx = user_to_idx[user_id]
        for item_id in items:
            matrix[user_idx, item_to_idx[item_id]] = 1.0
    if weighting == "tfidf":
        item_df = matrix.sum(axis=0)
        idf = np.log1p(matrix.shape[0] / np.maximum(item_df, 1.0))
        matrix = matrix * idf
        row_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(row_norm, 1e-12)
    elif weighting != "binary":
        raise ValueError(f"Unsupported MF weighting mode: {weighting}")
    effective_components = max(2, min(n_components, min(matrix.shape) - 1))
    svd = TruncatedSVD(
        n_components=effective_components,
        n_iter=max(5, n_iter),
        random_state=42,
        algorithm=algorithm,
        tol=max(0.0, float(tol)) if algorithm == "arpack" else 0.0,
    )
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
    batch_size: int = 4096,
    max_grad_norm: float = 1.0,
    early_stop_rounds: int = 0,
    early_stop_metric: str = "val_recall_at_k",
    early_stop_k: int = 20,
    early_stop_tolerance: float = 0.0,
    validation_interactions: dict[str, set[str]] | None = None,
    temperature: float = 1.0,
    normalize_embeddings: bool = True,
    tower_hidden_dim: int = 0,
    tower_dropout: float = 0.0,
    device: str = "auto",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    torch.manual_seed(42)

    if device != "auto":
        raise ValueError("device must be 'auto' in this project.")
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(42)
    if verbose:
        print(f"[two_tower] device={device_obj.type}")

    positives: list[tuple[int, int]] = []
    for user_id, items in train.items():
        user_idx = user_to_idx[user_id]
        for item_id in items:
            if item_id in item_to_idx:
                positives.append((user_idx, item_to_idx[item_id]))

    if not positives: raise ValueError("No positive interactions available for two-tower training.")

    user_embedding_layer = torch.nn.Embedding(len(user_to_idx), embedding_dim).to(device_obj)
    item_embedding_layer = torch.nn.Embedding(len(item_to_idx), embedding_dim).to(device_obj)
    user_tower = (
        nn.Sequential(
            nn.Linear(embedding_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(max(0.0, tower_dropout)),
            nn.Linear(tower_hidden_dim, embedding_dim),
        ).to(device_obj)
        if tower_hidden_dim > 0
        else nn.Identity()
    )
    item_tower = (
        nn.Sequential(
            nn.Linear(embedding_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(max(0.0, tower_dropout)),
            nn.Linear(tower_hidden_dim, embedding_dim),
        ).to(device_obj)
        if tower_hidden_dim > 0
        else nn.Identity()
    )
    with torch.no_grad():
        user_embedding_layer.weight.normal_(mean=0.0, std=0.05)
        item_embedding_layer.weight.normal_(mean=0.0, std=0.05)
    optimizer = torch.optim.AdamW(
        list(user_embedding_layer.parameters()) + list(item_embedding_layer.parameters()) + list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=learning_rate,
        weight_decay=l2_reg if l2_reg > 0 else 0.0,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if early_stop_metric not in {"loss", "val_recall_at_k"}:
        raise ValueError("early_stop_metric must be one of: loss, val_recall_at_k")

    validation_user_indices: list[int] = []
    validation_target_indices: list[set[int]] = []
    validation_seen_indices: list[list[int]] = []
    if validation_interactions:
        for user_id, items in validation_interactions.items():
            if user_id not in user_to_idx:
                continue
            item_indices = {item_to_idx[item_id] for item_id in items if item_id in item_to_idx}
            if item_indices:
                validation_user_indices.append(user_to_idx[user_id])
                validation_target_indices.append(item_indices)
                seen_train = train.get(user_id, set())
                validation_seen_indices.append([item_to_idx[item_id] for item_id in seen_train if item_id in item_to_idx])

    best_loss = float("inf")
    best_recall = -1.0
    no_improvement_rounds = 0
    def _validation_recall_at_k() -> float:
        if not validation_user_indices:
            return 0.0
        with torch.no_grad():
            user_tower.eval(); item_tower.eval()
            item_matrix = item_tower(item_embedding_layer.weight.detach())
            if normalize_embeddings:
                item_matrix = F.normalize(item_matrix, p=2, dim=1)
            k = min(max(1, early_stop_k), item_matrix.shape[0])
            batch_eval_size = max(256, min(2048, batch_size))
            recall_sum = 0.0
            for start in range(0, len(validation_user_indices), batch_eval_size):
                end = min(start + batch_eval_size, len(validation_user_indices))
                user_vec = user_tower(user_embedding_layer.weight[validation_user_indices[start:end]].detach())
                if normalize_embeddings:
                    user_vec = F.normalize(user_vec, p=2, dim=1)
                scores = (user_vec @ item_matrix.T) / float(temperature)
                for row_idx, seen_idx in enumerate(validation_seen_indices[start:end]):
                    if seen_idx:
                        scores[row_idx, seen_idx] = -float("inf")
                batch_topk = torch.topk(scores, k=k, dim=1).indices.cpu().tolist()
                for row_idx, ranked in enumerate(batch_topk):
                    target_indices = validation_target_indices[start + row_idx]
                    hits = len(target_indices.intersection(ranked))
                    recall_sum += hits / max(1, len(target_indices))
            user_tower.train(); item_tower.train()
            return recall_sum / len(validation_user_indices)

    for epoch in range(1, epochs + 1):
        user_tower.train(); item_tower.train()
        rng.shuffle(positives)
        positive_array = np.asarray(positives, dtype=np.int64)
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for start in range(0, len(positive_array), max(1, batch_size)):
            batch_pos = positive_array[start : start + max(1, batch_size)]
            if batch_pos.size == 0:
                continue
            batch_users = torch.from_numpy(batch_pos[:, 0]).long().to(device_obj)
            batch_items = torch.from_numpy(batch_pos[:, 1]).long().to(device_obj)
            user_vec = user_tower(user_embedding_layer(batch_users))
            item_vec = item_tower(item_embedding_layer(batch_items))
            if normalize_embeddings:
                user_vec = F.normalize(user_vec, p=2, dim=1)
                item_vec = F.normalize(item_vec, p=2, dim=1)
            # In-batch negatives: each other positive item in batch acts as a negative for this user.
            logits = (user_vec @ item_vec.T) / float(temperature)
            if negative_samples > 0:
                neg_items = torch.from_numpy(np_rng.integers(0, len(item_to_idx), size=(len(batch_users), negative_samples), dtype=np.int64)).long().to(device_obj)
                neg_vec = item_tower(item_embedding_layer(neg_items))
                if normalize_embeddings:
                    neg_vec = F.normalize(neg_vec, p=2, dim=2)
                neg_logits = (user_vec.unsqueeze(1) * neg_vec).sum(dim=2) / float(temperature)
                logits = torch.cat([logits, neg_logits], dim=1)
            labels = torch.arange(len(batch_users), dtype=torch.long, device=device_obj)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(user_embedding_layer.parameters()) + list(item_embedding_layer.parameters()) + list(user_tower.parameters()) + list(item_tower.parameters()),
                    max_norm=max_grad_norm,
                )
            optimizer.step()
            epoch_loss_sum += float(loss.item())
            epoch_steps += 1
        if epoch_steps == 0:
            continue
        epoch_avg_loss = epoch_loss_sum / epoch_steps
        epoch_val_recall = _validation_recall_at_k() if validation_user_indices else 0.0
        tolerance = max(0.0, early_stop_tolerance)
        if early_stop_metric == "val_recall_at_k" and validation_user_indices:
            improved = (epoch_val_recall - best_recall) > tolerance
            if improved:
                best_recall = epoch_val_recall
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
        else:
            improved = (best_loss - epoch_avg_loss) > tolerance
            if improved:
                best_loss = epoch_avg_loss
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
        if verbose:
            print(f"[two_tower] epoch={epoch}/{epochs} avg_loss={epoch_avg_loss:.6f} val_recall@{early_stop_k}={epoch_val_recall:.6f}")
        if early_stop_rounds > 0 and no_improvement_rounds >= early_stop_rounds:
            if verbose:
                print(
                    "[two_tower] early-stop:",
                    f"no {early_stop_metric} improvement > {max(0.0, early_stop_tolerance):.6f}",
                    f"for {early_stop_rounds} round(s)",
                )
            break

    with torch.no_grad():
        user_tower.eval(); item_tower.eval()
        final_user = user_tower(user_embedding_layer.weight.detach()).cpu().numpy()
        final_item = item_tower(item_embedding_layer.weight.detach()).cpu().numpy()
    return final_user, final_item
