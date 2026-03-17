"""Two-tower recommender model module.

Workflow Steps:
1) Read the shared user->items train interactions.
2) Train embedding-only two-tower user/item embeddings with contrastive learning.
3) Apply optional row-wise L2 normalization to align retrieval scoring.
4) Return user/item embedding matrices for shared evaluation/selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mle_marketplace_growth.recommender.constants import DEVICE, EARLY_STOP_METRIC, NORMALIZE_EMBEDDINGS


@dataclass(frozen=True)
class TwoTowerValidationCache:
    user_indices: list[int]
    target_item_indices: list[set[int]]
    seen_train_item_indices: list[list[int]]


@dataclass(frozen=True)
class TwoTowerScorer:
    """What: Score two-tower-model candidates for evaluation and prediction.
    Why: Keeps two-tower scoring behavior local to the two-tower module.
    """

    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    model_name: str = "two_tower"

    def score_candidate_indices(self, user_index: int, candidate_item_indices: list[int]) -> np.ndarray:
        return np.asarray(self.item_embeddings[candidate_item_indices].dot(self.user_embeddings[user_index]))

    def rank_user_topk(
        self,
        *,
        user_index: int,
        top_k: int,
        item_count: int,
        seen_indices: set[int],
        ann_index: faiss.Index | None,
    ) -> tuple[list[int], list[float]]:
        if len(seen_indices) >= item_count:
            return [], []
        if ann_index is None:
            raise ValueError("ANN index is required for two-tower scoring.")
        user_vector = self.user_embeddings[user_index].reshape(1, -1)
        oversample = min(item_count, max(top_k + len(seen_indices) + 20, top_k * 3))
        _, indices = ann_index.search(user_vector.astype(np.float32), oversample)
        ranked_item_indices: list[int] = []
        for item_idx in indices[0].tolist():
            if item_idx < 0 or item_idx in seen_indices:
                continue
            ranked_item_indices.append(int(item_idx))
            if len(ranked_item_indices) >= top_k:
                break
        ranked_scores = [float(self.item_embeddings[item_idx].dot(user_vector[0])) for item_idx in ranked_item_indices]
        return ranked_item_indices, ranked_scores

    def item_matrix(self) -> np.ndarray:
        return self.item_embeddings

    @classmethod
    def load_from_dir(cls, model_dir: Path) -> "TwoTowerScorer":
        """What: Load two-tower scorer artifacts from one model directory.
        Why: Keeps two-tower artifact-loading logic local to the two-tower model module.
        """
        return cls(
            user_embeddings=np.load(model_dir / "user_embeddings.npy"),
            item_embeddings=np.load(model_dir / "item_embeddings.npy"),
        )


class _PositivePairDataset(Dataset):
    """What: Expose positive (user_idx, item_idx) rows as a PyTorch dataset.
    Why: Lets the two-tower train loop use the standard Dataset/DataLoader batching pattern.
    """

    def __init__(self, positive_pairs: np.ndarray) -> None:
        self._positive_pairs = np.asarray(positive_pairs, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self._positive_pairs))

    def __getitem__(self, index: int) -> tuple[int, int]:
        user_idx, item_idx = self._positive_pairs[index]
        return int(user_idx), int(item_idx)


class UserEncoder(torch.nn.Module):
    """What: Map user indices to embedding vectors.
    Why: Makes the user tower explicit instead of accessing one raw embedding table directly.
    """

    def __init__(self, user_count: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(user_count, embedding_dim)
        with torch.no_grad():
            self.embedding.weight.normal_(mean=0.0, std=0.05)

    def forward(self, user_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(user_indices)


class ItemEncoder(torch.nn.Module):
    """What: Map item indices to embedding vectors.
    Why: Makes the item tower explicit instead of accessing one raw embedding table directly.
    """

    def __init__(self, item_count: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(item_count, embedding_dim)
        with torch.no_grad():
            self.embedding.weight.normal_(mean=0.0, std=0.05)

    def forward(self, item_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(item_indices)


class TwoTowerModel(torch.nn.Module):
    """What: Bundle user and item encoders into one explicit two-tower model.
    Why: Keeps the neural model definition clearer than managing raw embedding tables separately.
    """

    def __init__(self, user_count: int, item_count: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_encoder = UserEncoder(user_count, embedding_dim)
        self.item_encoder = ItemEncoder(item_count, embedding_dim)

    def user_embeddings(self, user_indices: torch.Tensor) -> torch.Tensor:
        return self.user_encoder(user_indices)

    def item_embeddings(self, item_indices: torch.Tensor) -> torch.Tensor:
        return self.item_encoder(item_indices)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """What: L2-normalize each row vector.
    Why: Keeps embedding similarity as cosine-style dot product.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _interaction_pairs(
    interactions: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> np.ndarray:
    """What: Convert user->items interactions into [N, 2] (user_idx, item_idx) pairs.
    Why: Gives the two-tower path one explicit positive-pair representation.
    """
    pairs = [
        (user_to_idx[user_id], item_to_idx[item_id])
        for user_id, items in interactions.items()
        for item_id in items
        if user_id in user_to_idx and item_id in item_to_idx
    ]
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _build_validation_cache(
    train: dict[str, set[str]],
    validation: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> TwoTowerValidationCache:
    """What: Build explicit validation lookup inputs for two-tower early stopping.
    Why: Keeps user/item split prep in the train module, not in the low-level model loop.
    """
    user_indices: list[int] = []
    target_item_indices: list[set[int]] = []
    seen_train_item_indices: list[list[int]] = []
    for user_id, items in validation.items():
        if user_id not in user_to_idx:
            continue
        item_indices = {item_to_idx[item_id] for item_id in items if item_id in item_to_idx}
        if not item_indices:
            continue
        user_indices.append(user_to_idx[user_id])
        target_item_indices.append(item_indices)
        seen_train = train.get(user_id, set())
        seen_train_item_indices.append([item_to_idx[item_id] for item_id in seen_train if item_id in item_to_idx])
    return TwoTowerValidationCache(
        user_indices=user_indices,
        target_item_indices=target_item_indices,
        seen_train_item_indices=seen_train_item_indices,
    )


def _validation_recall_at_k(
    *,
    model: TwoTowerModel,
    validation_cache: TwoTowerValidationCache,
    early_stop_k: int,
    batch_size: int,
    temperature: float,
    normalize_embeddings: bool,
) -> float:
    """What: Compute validation Recall@K for early stopping.
    Why: Keeps validation scoring logic separate from the epoch training loop.
    """
    if not validation_cache.user_indices:
        return 0.0
    with torch.no_grad():
        item_embedding_matrix = model.item_encoder.embedding.weight.detach()
        if normalize_embeddings:
            item_embedding_matrix = F.normalize(item_embedding_matrix, p=2, dim=1)
        k = min(max(1, early_stop_k), item_embedding_matrix.shape[0])
        batch_eval_size = max(256, min(2048, batch_size))
        recall_sum = 0.0
        for start in range(0, len(validation_cache.user_indices), batch_eval_size):
            end = min(start + batch_eval_size, len(validation_cache.user_indices))
            batch_user_indices = torch.as_tensor(
                validation_cache.user_indices[start:end],
                dtype=torch.long,
                device=item_embedding_matrix.device,
            )
            user_embeddings_batch = model.user_embeddings(batch_user_indices).detach()
            if normalize_embeddings:
                user_embeddings_batch = F.normalize(user_embeddings_batch, p=2, dim=1)
            similarity_scores = (user_embeddings_batch @ item_embedding_matrix.T) / float(temperature)
            for row_idx, seen_idx in enumerate(validation_cache.seen_train_item_indices[start:end]):
                if seen_idx:
                    similarity_scores[row_idx, seen_idx] = -float("inf")
            batch_topk = torch.topk(similarity_scores, k=k, dim=1).indices.cpu().tolist()
            for row_idx, ranked in enumerate(batch_topk):
                target_indices = validation_cache.target_item_indices[start + row_idx]
                hits = len(target_indices.intersection(ranked))
                recall_sum += hits / max(1, len(target_indices))
        return recall_sum / len(validation_cache.user_indices)


def _train_two_tower_from_pairs(
    *,
    positive_array: np.ndarray,
    user_count: int,
    item_count: int,
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
    validation_user_indices: list[int] | None = None,
    validation_target_indices: list[set[int]] | None = None,
    validation_seen_indices: list[list[int]] | None = None,
    temperature: float = 1.0,
    normalize_embeddings: bool = True,
    device: str = "auto",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """What: Train two-tower retrieval embeddings from explicit pair/cached-index inputs.
    Why: Keeps the low-level two-tower training loop in the model-specific module.
    """
    def _build_training_loader() -> DataLoader:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(42)
        return DataLoader(
            _PositivePairDataset(positive_array),
            batch_size=max(1, batch_size),
            shuffle=True,
            drop_last=False,
            generator=loader_generator,
        )

    def _init_two_tower_components() -> tuple[TwoTowerModel, torch.optim.Optimizer, torch.nn.Module]:
        model = TwoTowerModel(user_count=user_count, item_count=item_count, embedding_dim=embedding_dim).to(device_obj)
        optimizer_obj = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=l2_reg if l2_reg > 0 else 0.0,
        )
        loss_module = torch.nn.CrossEntropyLoss()
        return model, optimizer_obj, loss_module

    np_rng = np.random.default_rng(42)
    torch.manual_seed(42)
    if device != "auto":
        raise ValueError("device must be 'auto' in this project.")
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(42)
    if verbose:
        print(f"[two_tower] device={device_obj.type}")
    if positive_array.size == 0:
        raise ValueError("No positive interactions available for two-tower training.")
    training_loader = _build_training_loader()
    model, optimizer, loss_fn = _init_two_tower_components()
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if early_stop_metric not in {"loss", "val_recall_at_k"}:
        raise ValueError("early_stop_metric must be one of: loss, val_recall_at_k")

    validation_cache = TwoTowerValidationCache(
        validation_user_indices or [],
        validation_target_indices or [],
        validation_seen_indices or [],
    )
    best_loss = float("inf")
    best_recall = -1.0
    no_improvement_rounds = 0

    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for batch_users_cpu, batch_positive_items_cpu in training_loader:
            if len(batch_users_cpu) == 0:
                continue
            batch_users = batch_users_cpu.long().to(device_obj)
            batch_positive_items = batch_positive_items_cpu.long().to(device_obj)

            # Score positive user-item pairs inside the current mini-batch.
            user_embeddings_batch = model.user_embeddings(batch_users)
            positive_item_embeddings_batch = model.item_embeddings(batch_positive_items)
            if normalize_embeddings:
                user_embeddings_batch = F.normalize(user_embeddings_batch, p=2, dim=1)
                positive_item_embeddings_batch = F.normalize(positive_item_embeddings_batch, p=2, dim=1)
            similarity_logits = (user_embeddings_batch @ positive_item_embeddings_batch.T) / float(temperature)

            # Append sampled negatives if configured.
            if negative_samples > 0:
                sampled_negative_items = torch.from_numpy(
                    np_rng.integers(0, item_count, size=(len(batch_users), negative_samples), dtype=np.int64)
                ).long().to(device_obj)
                negative_item_embeddings_batch = model.item_embeddings(sampled_negative_items)
                if normalize_embeddings:
                    negative_item_embeddings_batch = F.normalize(negative_item_embeddings_batch, p=2, dim=2)
                sampled_negative_logits = (
                    (user_embeddings_batch.unsqueeze(1) * negative_item_embeddings_batch).sum(dim=2) / float(temperature)
                )
                similarity_logits = torch.cat([similarity_logits, sampled_negative_logits], dim=1)

            # Optimize one contrastive training step.
            positive_target_indices = torch.arange(len(batch_users), dtype=torch.long, device=device_obj)
            loss = loss_fn(similarity_logits, positive_target_indices)
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            epoch_loss_sum += float(loss.item())
            epoch_steps += 1
        if epoch_steps == 0:
            continue

        # Evaluate the epoch and update early-stopping state.
        epoch_avg_loss = epoch_loss_sum / epoch_steps
        epoch_val_recall = (
            _validation_recall_at_k(
                model=model,
                validation_cache=validation_cache,
                early_stop_k=early_stop_k,
                batch_size=batch_size,
                temperature=temperature,
                normalize_embeddings=normalize_embeddings,
            )
            if validation_cache.user_indices
            else 0.0
        )
        tolerance = max(0.0, early_stop_tolerance)
        if early_stop_metric == "val_recall_at_k" and validation_cache.user_indices:
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
        final_user = model.user_encoder.embedding.weight.detach().cpu().numpy()
        final_item = model.item_encoder.embedding.weight.detach().cpu().numpy()
    return final_user, final_item


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
    device: str = "auto",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """What: Backward-compatible wrapper for tests and existing call sites.
    Why: Keeps the public helper name stable while the model-specific module owns the real training loop.
    """
    positive_array = _interaction_pairs(train, user_to_idx, item_to_idx)
    validation_cache = (
        _build_validation_cache(train, validation_interactions, user_to_idx, item_to_idx)
        if validation_interactions
        else TwoTowerValidationCache([], [], [])
    )
    return _train_two_tower_from_pairs(
        positive_array=positive_array,
        user_count=len(user_to_idx),
        item_count=len(item_to_idx),
        embedding_dim=embedding_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        negative_samples=negative_samples,
        l2_reg=l2_reg,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        early_stop_rounds=early_stop_rounds,
        early_stop_metric=early_stop_metric,
        early_stop_k=early_stop_k,
        early_stop_tolerance=early_stop_tolerance,
        validation_user_indices=validation_cache.user_indices,
        validation_target_indices=validation_cache.target_item_indices,
        validation_seen_indices=validation_cache.seen_train_item_indices,
        temperature=temperature,
        normalize_embeddings=normalize_embeddings,
        device=device,
        verbose=verbose,
    )


def train_two_tower_candidate(
    train: dict[str, set[str]],
    validation: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    *,
    embedding_dim: int,
    epochs: int,
    learning_rate: float,
    negative_samples: int,
    batch_size: int,
    l2_reg: float,
    max_grad_norm: float,
    early_stop_rounds: int,
    early_stop_k: int,
    early_stop_tolerance: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """What: Train the two-tower candidate user/item embeddings.
    Why: Keeps two-tower-specific fitting and knobs separate from other models.
    """
    positive_train_pairs = _interaction_pairs(train, user_to_idx, item_to_idx)
    validation_cache = _build_validation_cache(train, validation, user_to_idx, item_to_idx)
    tt_user_embeddings, tt_item_embeddings = _train_two_tower_from_pairs(
        positive_array=positive_train_pairs,
        user_count=len(user_to_idx),
        item_count=len(item_to_idx),
        embedding_dim=embedding_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        negative_samples=negative_samples,
        batch_size=batch_size,
        l2_reg=l2_reg,
        max_grad_norm=max_grad_norm,
        early_stop_rounds=early_stop_rounds,
        early_stop_metric=EARLY_STOP_METRIC,
        early_stop_k=early_stop_k,
        early_stop_tolerance=early_stop_tolerance,
        validation_user_indices=validation_cache.user_indices,
        validation_target_indices=validation_cache.target_item_indices,
        validation_seen_indices=validation_cache.seen_train_item_indices,
        temperature=temperature,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        device=DEVICE,
        verbose=True,
    )
    if NORMALIZE_EMBEDDINGS:
        tt_user_embeddings = _l2_normalize_rows(tt_user_embeddings)
        tt_item_embeddings = _l2_normalize_rows(tt_item_embeddings)
    print("[recommender.models.two_tower] trained two_tower")
    return tt_user_embeddings, tt_item_embeddings
