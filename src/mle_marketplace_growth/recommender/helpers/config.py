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
    splits_path: Path
    user_index_path: Path
    item_index_path: Path
    artifacts_dir: Path
    top_k: int


@dataclass(frozen=True)
class RecommenderArtifactPaths:
    model_bundle: Path
    topk_recommendations: Path
    output_validation_summary: Path
    output_interpretation: Path


def load_recommender_runtime_config(config_path: str) -> RecommenderRuntimeConfig:
    """What: Load runtime config fields used across recommender scripts.
    Why: Centralizes required key parsing for train/predict/pipeline consistency.
    """
    cfg = load_yaml_defaults(config_path, "Recommender config")
    artifacts_root = Path(str(cfg_required(cfg, "artifacts_dir")))
    as_of_date = str(cfg_required(cfg, "recommender_max_event_date"))
    return RecommenderRuntimeConfig(
        cfg=cfg,
        splits_path=Path(str(cfg_required(cfg, "splits_path"))),
        user_index_path=Path(str(cfg_required(cfg, "user_index_path"))),
        item_index_path=Path(str(cfg_required(cfg, "item_index_path"))),
        artifacts_dir=artifacts_root / f"as_of={as_of_date}",
        top_k=int(cfg_required(cfg, "top_k")),
    )


def artifact_paths(runtime: RecommenderRuntimeConfig) -> RecommenderArtifactPaths:
    """What: Derive canonical artifact file paths from runtime config.
    Why: Keeps all script outputs under a stable, shared run folder layout.
    """
    return RecommenderArtifactPaths(
        model_bundle=runtime.artifacts_dir / "model_bundle.pkl",
        topk_recommendations=runtime.artifacts_dir / "topk_recommendations.csv",
        output_validation_summary=runtime.artifacts_dir / "output_validation_summary.json",
        output_interpretation=runtime.artifacts_dir / "output_interpretation.md",
    )
