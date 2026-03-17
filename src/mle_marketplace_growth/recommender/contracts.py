from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mle_marketplace_growth.recommender.helpers.data import EntityIndex, SplitInteractions


@dataclass(frozen=True)
class RecommenderTrainParams:
    embedding_dim: int
    epochs: int
    learning_rate: float
    negative_samples: int
    batch_size: int
    l2_reg: float
    max_grad_norm: float
    early_stop_rounds: int
    early_stop_k: int
    early_stop_tolerance: float
    temperature: float
    mf_components: int
    mf_n_iter: int
    mf_weighting: str


@dataclass(frozen=True)
class CandidateModelArtifacts:
    artifacts_by_model: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class TrainingInputs:
    split_interactions: SplitInteractions
    user_index: EntityIndex
    item_index: EntityIndex


@dataclass(frozen=True)
class ModelSelectionResult:
    validation_metrics: list[dict]
    test_metrics: list[dict]
    selected_model_name: str
