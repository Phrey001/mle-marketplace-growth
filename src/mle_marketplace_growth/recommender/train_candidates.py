"""Train recommender candidate models on one shared train/validation split.

Workflow Steps:
1) Accept shared split interactions, entity indices, and train params.
2) Train popularity, MF, and two-tower candidates in dedicated model modules.
3) Return model-scoped in-memory artifacts for downstream offline selection.
"""

from __future__ import annotations

from mle_marketplace_growth.recommender.contracts import (
    CandidateModelArtifacts,
    RecommenderTrainParams,
)
from mle_marketplace_growth.recommender.helpers.data import EntityIndex, SplitInteractions
from mle_marketplace_growth.recommender.models.mf import train_mf_candidate
from mle_marketplace_growth.recommender.models.popularity import build_item_interaction_counts, train_popularity_candidate
from mle_marketplace_growth.recommender.models.two_tower import train_two_tower_candidate


def train_candidate_models(
    *,
    split_interactions: SplitInteractions,
    user_index: EntityIndex,
    item_index: EntityIndex,
    train_params: RecommenderTrainParams,
) -> CandidateModelArtifacts:
    """What: Train popularity, MF, and two-tower candidates through dedicated model modules.
    Why: This stage intentionally trains all supported model families in one run so they are compared fairly on the same data split.
    """
    train_interactions = split_interactions.train
    validation_interactions = split_interactions.validation
    user_id_to_idx = user_index.id_to_idx
    item_id_to_idx = item_index.id_to_idx

    # Popularity only needs item-level counts after train interactions are fixed.
    popularity_item_counts = build_item_interaction_counts(train_interactions, item_id_to_idx)
    popularity_scores = train_popularity_candidate(popularity_item_counts)
    mf_user_embeddings, mf_item_embeddings = train_mf_candidate(
        train_interactions,
        user_id_to_idx,
        item_id_to_idx,
        mf_components=train_params.mf_components,
        mf_n_iter=train_params.mf_n_iter,
        mf_weighting=train_params.mf_weighting,
    )
    tt_user_embeddings, tt_item_embeddings = train_two_tower_candidate(
        train_interactions,
        validation_interactions,
        user_id_to_idx,
        item_id_to_idx,
        embedding_dim=train_params.embedding_dim,
        epochs=train_params.epochs,
        learning_rate=train_params.learning_rate,
        negative_samples=train_params.negative_samples,
        batch_size=train_params.batch_size,
        l2_reg=train_params.l2_reg,
        max_grad_norm=train_params.max_grad_norm,
        early_stop_rounds=train_params.early_stop_rounds,
        early_stop_k=train_params.early_stop_k,
        early_stop_tolerance=train_params.early_stop_tolerance,
        temperature=train_params.temperature,
    )
    return CandidateModelArtifacts(
        artifacts_by_model={
            "popularity": {"scores": popularity_scores},
            "mf": {
                "user_embeddings": mf_user_embeddings,
                "item_embeddings": mf_item_embeddings,
            },
            "two_tower": {
                "user_embeddings": tt_user_embeddings,
                "item_embeddings": tt_item_embeddings,
            },
        },
    )
