"""Matrix-factorization recommender model module.

Workflow Steps:
1) Convert train interactions into numeric (user_idx, item_idx) pairs.
2) Build the user-item interaction matrix inside the MF helper.
3) Apply the configured MF weighting policy.
4) Fit truncated-SVD user/item factors for shared evaluation/selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from mle_marketplace_growth.recommender.constants import MF_ALGORITHM


def _interaction_pairs(
    interactions: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> np.ndarray:
    """What: Convert user->items interactions into [N, 2] (user_idx, item_idx) pairs.
    Why: Gives MF training one simple numeric input type before sparse-matrix construction.
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


def _build_mf_interaction_matrix(
    train: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> sp.csr_matrix:
    """What: Build the MF user-item interaction matrix in sparse CSR form.
    Why: Keeps the MF path close to the canonical matrix-factorization representation.
    """
    train_pairs = _interaction_pairs(train, user_to_idx, item_to_idx)
    if train_pairs.size == 0:
        return sp.csr_matrix((len(user_to_idx), len(item_to_idx)), dtype=float)
    user_indices, item_indices = train_pairs.T
    data = np.ones(len(train_pairs), dtype=float)
    return sp.csr_matrix(
        (data, (user_indices, item_indices)),
        shape=(len(user_to_idx), len(item_to_idx)),
        dtype=float,
    )


def _train_mf(
    interaction_matrix: sp.csr_matrix,
    n_components: int,
    n_iter: int = 15,
    weighting: str = "tfidf",
    algorithm: str = "randomized",
    tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """What: Train MF-style user/item factors via truncated SVD.
    Why: Provides a fast latent-factor baseline for retrieval comparison.
    """
    matrix = interaction_matrix.copy().astype(float)
    if weighting == "tfidf":
        item_df = np.asarray(matrix.sum(axis=0)).ravel()
        idf = np.log1p(matrix.shape[0] / np.maximum(item_df, 1.0))
        matrix = matrix.multiply(idf)
        matrix = normalize(matrix, norm="l2", axis=1, copy=False)
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


@dataclass(frozen=True)
class MFScorer:
    """What: Score MF-model candidates for evaluation and prediction.
    Why: Keeps MF scoring behavior local to the MF module.
    """

    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    model_name: str = "mf"

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
            raise ValueError("ANN index is required for MF scoring.")
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
    def load_from_dir(cls, model_dir: Path) -> "MFScorer":
        """What: Load MF scorer artifacts from one model directory.
        Why: Keeps MF artifact-loading logic local to the MF model module.
        """
        return cls(
            user_embeddings=np.load(model_dir / "user_embeddings.npy"),
            item_embeddings=np.load(model_dir / "item_embeddings.npy"),
        )


def train_mf_candidate(
    train: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    *,
    mf_components: int,
    mf_n_iter: int,
    mf_weighting: str,
) -> tuple[np.ndarray, np.ndarray]:
    """What: Train the MF candidate user/item factors.
    Why: Keeps MF-specific fitting and MF-only knobs separate from other models.
    """
    interaction_matrix = _build_mf_interaction_matrix(train, user_to_idx, item_to_idx)
    mf_user_embeddings, mf_item_embeddings = _train_mf(
        interaction_matrix,
        n_components=mf_components,
        n_iter=mf_n_iter,
        weighting=mf_weighting,
        algorithm=MF_ALGORITHM,
        tol=0.0,
    )
    print(
        "[recommender.models.mf] trained mf baseline "
        f"(components={mf_components}, n_iter={mf_n_iter}, weighting={mf_weighting}, algorithm={MF_ALGORITHM})"
    )
    return mf_user_embeddings, mf_item_embeddings
