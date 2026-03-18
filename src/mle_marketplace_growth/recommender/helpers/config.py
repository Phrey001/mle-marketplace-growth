from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mle_marketplace_growth.helpers import cfg_required, load_yaml_defaults

"""Config helpers for recommender runtime path and key resolution.

Workflow Steps:
1) Load YAML defaults and required fields once.
2) Derive deterministic run folder keyed by max event date.
3) Expose typed runtime/path objects shared by train/predict/validate.
"""


@dataclass(frozen=True)
class RecommenderRuntimeConfig:
    cfg: dict
    user_item_splits_path: Path
    user_index_path: Path
    item_index_path: Path
    artifacts_dir: Path
    top_k: int


@dataclass(frozen=True)
class RecommenderArtifactPaths:
    offline_eval_dir: Path
    serving_dir: Path
    report_dir: Path
    selected_model_meta: Path
    shared_context: Path
    train_metrics: Path
    validation_retrieval_metrics: Path
    test_retrieval_metrics: Path
    ann_index: Path
    ann_index_meta: Path
    item_embeddings: Path
    item_embedding_index: Path
    topk_recommendations: Path
    output_validation_summary: Path
    output_interpretation: Path


def _recommender_gold_root() -> Path:
    """What: Return the fixed recommender gold feature-store root.
    Why: Recommender ML reads a stable upstream contract under data/gold/feature_store/recommender.
    """
    return Path("data") / "gold" / "feature_store" / "recommender"


def _recommender_runtime_root(as_of_date: str) -> Path:
    """What: Return the fixed recommender artifact root for one as-of date.
    Why: Keeps run outputs in one deterministic folder without YAML path knobs.
    """
    return Path("artifacts") / "recommender" / f"as_of={as_of_date}"


def load_recommender_runtime_config(config_path: str) -> RecommenderRuntimeConfig:
    """What: Load runtime config fields used across recommender scripts.
    Why: Centralizes required key parsing for train/predict/pipeline consistency.
    """
    cfg = load_yaml_defaults(config_path, "Recommender config")
    as_of_date = str(cfg_required(cfg, "recommender_max_event_date"))
    gold_root = _recommender_gold_root()
    return RecommenderRuntimeConfig(
        cfg=cfg,
        user_item_splits_path=gold_root / "user_item_splits" / "user_item_splits.parquet",
        user_index_path=gold_root / "user_index" / "user_index.parquet",
        item_index_path=gold_root / "item_index" / "item_index.parquet",
        artifacts_dir=_recommender_runtime_root(as_of_date),
        top_k=int(cfg_required(cfg, "top_k")),
    )


def artifact_paths(runtime: RecommenderRuntimeConfig) -> RecommenderArtifactPaths:
    """What: Derive canonical artifact file paths from runtime config.
    Why: Keeps all script outputs under a stable, shared run folder layout.
    """
    offline_eval_dir = runtime.artifacts_dir / "offline_eval"
    serving_dir = runtime.artifacts_dir / "serving"
    report_dir = runtime.artifacts_dir / "report"
    return RecommenderArtifactPaths(
        offline_eval_dir=offline_eval_dir,
        serving_dir=serving_dir,
        report_dir=report_dir,
        selected_model_meta=offline_eval_dir / "selected_model_meta.json",
        shared_context=offline_eval_dir / "shared_context.json",
        train_metrics=offline_eval_dir / "train_metrics.json",
        validation_retrieval_metrics=offline_eval_dir / "validation_retrieval_metrics.json",
        test_retrieval_metrics=offline_eval_dir / "test_retrieval_metrics.json",
        ann_index=serving_dir / "ann_index.bin",
        ann_index_meta=serving_dir / "ann_index_meta.json",
        item_embeddings=serving_dir / "item_embeddings.npy",
        item_embedding_index=serving_dir / "item_embedding_index.json",
        topk_recommendations=serving_dir / "topk_recommendations.csv",
        output_validation_summary=report_dir / "output_validation_summary.json",
        output_interpretation=report_dir / "output_interpretation.md",
    )
