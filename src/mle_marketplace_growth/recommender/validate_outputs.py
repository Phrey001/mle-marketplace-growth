"""Validate recommender artifacts and write a concise interpretation summary.

Workflow Steps:
1) Resolve canonical artifact paths for one recommender run.
2) Verify required files exist and load metric/index payloads.
3) Run contract checks (selected model, metric bounds, ANN artifacts, Top-K rows).
4) Write machine-readable validation summary JSON.
5) Write human-readable interpretation markdown.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from mle_marketplace_growth.helpers import read_json, write_json
from mle_marketplace_growth.recommender.helpers.config import artifact_paths, load_recommender_runtime_config
from mle_marketplace_growth.recommender.constants import ANN_BACKEND, EXPECTED_MODELS
from mle_marketplace_growth.recommender.helpers.artifacts import (
    RetrievalMetricsPayload,
    _load_retrieval_metrics_payload,
    _load_selected_model_meta,
    _load_shared_runtime_context,
)


@dataclass(frozen=True)
class CoreArtifactPaths:
    train_metrics: Path
    validation_metrics: Path
    test_metrics: Path
    selected_model_meta: Path
    shared_context: Path
    index_json: Path
    ann_meta: Path
    topk_csv: Path


@dataclass(frozen=True)
class ValidationCheck:
    check: str
    description: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationSummary:
    passed: bool
    artifacts_dir: str
    checks: list[ValidationCheck]


def _count_csv_rows(path: Path) -> int:
    """What: Count data rows in a CSV artifact (excluding header).
    Why: Supports non-empty output checks without loading full payload into memory.
    """
    with path.open("r", encoding="utf-8", newline="") as file:
        return len(list(csv.DictReader(file)))


def _core_artifact_paths(artifacts_dir: Path) -> CoreArtifactPaths:
    """What: Build canonical recommender artifact file paths for one run root.
    Why: Keeps validation and interpretation readers aligned to the shared subfolder contract.
    """
    offline_eval_dir = artifacts_dir / "offline_eval"
    serving_dir = artifacts_dir / "serving"
    return CoreArtifactPaths(
        train_metrics=offline_eval_dir / "train_metrics.json",
        validation_metrics=offline_eval_dir / "validation_retrieval_metrics.json",
        test_metrics=offline_eval_dir / "test_retrieval_metrics.json",
        selected_model_meta=offline_eval_dir / "selected_model_meta.json",
        shared_context=offline_eval_dir / "shared_context.json",
        index_json=serving_dir / "item_embedding_index.json",
        ann_meta=serving_dir / "ann_index_meta.json",
        topk_csv=serving_dir / "topk_recommendations.csv",
    )


def _load_core_artifacts(artifacts_dir: Path) -> tuple[dict, RetrievalMetricsPayload, RetrievalMetricsPayload]:
    """What: Load train/validation/test metrics JSON artifacts.
    Why: Shared loader avoids duplicated JSON-read logic across report steps.
    """
    paths = _core_artifact_paths(artifacts_dir)
    return (
        read_json(paths.train_metrics),
        _load_retrieval_metrics_payload(read_json(paths.validation_metrics)),
        _load_retrieval_metrics_payload(read_json(paths.test_metrics)),
    )


def _validation_summary_payload(summary: ValidationSummary) -> dict:
    """What: Convert typed validation summary into the persisted JSON shape.
    Why: Keeps validation-output formatting centralized and explicit.
    """
    return {
        "passed": summary.passed,
        "artifacts_dir": summary.artifacts_dir,
        "checks": [
            {
                "check": row.check,
                "description": row.description,
                "passed": row.passed,
                "detail": row.detail,
            }
            for row in summary.checks
        ],
    }


def run_validation(artifacts_dir: Path, output_json: Path | None = None) -> tuple[bool, ValidationSummary]:
    """What: Validate recommender output artifacts against contract checks.
    Why: Provides a deterministic pass/fail gate before report consumption.
    """
    # ===== Load Inputs =====
    resolved_paths = _core_artifact_paths(artifacts_dir)
    for path in (
        resolved_paths.train_metrics,
        resolved_paths.validation_metrics,
        resolved_paths.test_metrics,
        resolved_paths.selected_model_meta,
        resolved_paths.shared_context,
        resolved_paths.index_json,
        resolved_paths.ann_meta,
        resolved_paths.topk_csv,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")
    train_metrics, validation_metrics, test_metrics = (
        read_json(resolved_paths.train_metrics),
        _load_retrieval_metrics_payload(read_json(resolved_paths.validation_metrics)),
        _load_retrieval_metrics_payload(read_json(resolved_paths.test_metrics)),
    )
    selected_model_meta = _load_selected_model_meta(read_json(resolved_paths.selected_model_meta))
    index_json = read_json(resolved_paths.index_json)
    ann_meta = read_json(resolved_paths.ann_meta)
    topk_count = _count_csv_rows(resolved_paths.topk_csv)

    validation_rows = validation_metrics.rows
    test_rows = test_metrics.rows
    selected_model = train_metrics.get("selected_model_name")
    expected_models = EXPECTED_MODELS

    # ===== Run Checks =====
    checks = [
        ValidationCheck(
            check="selected_model_present",
            description="Selected model must be one of popularity/mf/two_tower.",
            passed=selected_model in expected_models,
            detail=f"selected_model_name={selected_model}",
        ),
        ValidationCheck(
            check="baseline_rows_present",
            description="Validation and test metrics should contain popularity, mf, and two_tower rows.",
            passed={row.get("model_name") for row in validation_rows} == expected_models
            and {row.get("model_name") for row in test_rows} == expected_models,
            detail=f"validation_rows={len(validation_rows)}, test_rows={len(test_rows)}",
        ),
    ]
    # Metric bounds checks.
    metric_in_bounds = True
    for payload in [validation_rows, test_rows]:
        for row in payload:
            for key, value in row.get("metrics", {}).items():
                if key.startswith(("Recall@", "NDCG@", "HitRate@")) and not (0.0 <= float(value) <= 1.0):
                    metric_in_bounds = False
    checks.append(
        ValidationCheck(
            check="metrics_in_bounds",
            description="Recall/NDCG/HitRate metrics should stay in [0,1].",
            passed=metric_in_bounds,
            detail="all_metrics_checked",
        )
    )
    # Output completeness checks.
    checks.append(
        ValidationCheck(
            check="topk_recommendations_non_empty",
            description="Top-K recommendation output must have at least one row.",
            passed=topk_count > 0,
            detail=f"row_count={topk_count}",
        )
    )
    checks.append(
        ValidationCheck(
            check="selected_model_meta_matches_selection",
            description="Selected-model metadata should match the train selected model.",
            passed=selected_model_meta.selected_model_name == selected_model,
            detail=f"meta_selected={selected_model_meta.selected_model_name}",
        )
    )
    checks.append(
        ValidationCheck(
            check="embedding_index_model_matches_selection",
            description="Embedding index selected model should match train selected model.",
            passed=index_json.get("selected_model_name") == selected_model,
            detail=f"index_selected={index_json.get('selected_model_name')}",
        )
    )
    checks.append(
        ValidationCheck(
            check="ann_index_artifacts_present",
            description="ANN metadata and FAISS ANN index artifact must be present.",
            passed=(artifacts_dir / "serving" / "ann_index.bin").exists() and ann_meta.get("backend") == ANN_BACKEND,
            detail=f"ann_backend={ann_meta.get('backend')}",
        )
    )

    passed = all(check.passed for check in checks)
    summary = ValidationSummary(passed=passed, artifacts_dir=str(artifacts_dir), checks=checks)
    # ===== Write Outputs =====
    if output_json is not None:
        write_json(output_json, _validation_summary_payload(summary))
    return passed, summary


def write_interpretation(artifacts_dir: Path, output_md: Path | None = None) -> Path:
    """What: Write a concise markdown interpretation from validated artifacts.
    Why: Gives a fast human-readable summary for review and reporting.
    """
    # ===== Load Inputs =====
    core_paths = _core_artifact_paths(artifacts_dir)
    train_metrics, validation_metrics, test_metrics = _load_core_artifacts(artifacts_dir)
    shared_runtime_context = _load_shared_runtime_context(read_json(core_paths.shared_context))
    selected_model = train_metrics.get("selected_model_name", "unknown")
    selection_rule = train_metrics.get("selection_rule", "")
    catalog_size = len(shared_runtime_context.item_ids)
    anchor_k = int(train_metrics.get("k_value", 20))
    random_anchor = (anchor_k / catalog_size) if catalog_size > 0 else 0.0

    val_by_model = {row["model_name"]: row.get("metrics", {}) for row in validation_metrics.rows}
    test_by_model = {row["model_name"]: row.get("metrics", {}) for row in test_metrics.rows}
    def _lift(observed: float) -> str:
        if random_anchor <= 0:
            return "n/a"
        return f"{(observed / random_anchor):.2f}x"

    # ===== Build Summary =====
    val_recall = {model: float(metrics.get(f"Recall@{anchor_k}", 0.0)) for model, metrics in val_by_model.items()}
    test_recall = {model: float(metrics.get(f"Recall@{anchor_k}", 0.0)) for model, metrics in test_by_model.items()}
    lines = [
        "# Recommender Automated Interpretation",
        "",
        "## Model Selection",
        f"- Selected model: `{selected_model}`",
        f"- Selection rule: `{selection_rule}`",
        "",
        "## Random Baseline Anchor",
        f"- Catalog size (N): {catalog_size}",
        f"- Recommendation depth (K): {anchor_k}",
        f"- Random Recall@{anchor_k} anchor (K/N): {random_anchor:.6f}",
        "",
        "## Validation Snapshot",
        f"- Two-tower Recall@{anchor_k}: {val_recall.get('two_tower', 0.0):.4f} (lift vs random anchor: {_lift(val_recall.get('two_tower', 0.0))})",
        f"- MF Recall@{anchor_k}: {val_recall.get('mf', 0.0):.4f} (lift vs random anchor: {_lift(val_recall.get('mf', 0.0))})",
        f"- Popularity Recall@{anchor_k}: {val_recall.get('popularity', 0.0):.4f} (lift vs random anchor: {_lift(val_recall.get('popularity', 0.0))})",
        "",
        "## Test Snapshot",
        f"- Two-tower Recall@{anchor_k}: {test_recall.get('two_tower', 0.0):.4f} (lift vs random anchor: {_lift(test_recall.get('two_tower', 0.0))})",
        f"- MF Recall@{anchor_k}: {test_recall.get('mf', 0.0):.4f} (lift vs random anchor: {_lift(test_recall.get('mf', 0.0))})",
        f"- Popularity Recall@{anchor_k}: {test_recall.get('popularity', 0.0):.4f} (lift vs random anchor: {_lift(test_recall.get('popularity', 0.0))})",
        "",
        "_Scope: offline ranking quality only; causal business lift needs online experiment._",
        "",
    ]
    report_path = output_md or (artifacts_dir / "report" / "output_interpretation.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_validate_outputs(
    artifacts_dir: Path,
    *,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> tuple[ValidationSummary, Path]:
    """What: Run artifact validation and write the interpretation summary together.
    Why: Gives both the CLI and pipeline one validation/report entrypoint.
    """
    passed, summary = run_validation(artifacts_dir, output_json=output_json)
    if not passed:
        raise ValueError(
            f"Automated artifact validation failed: {[row for row in summary.checks if not row.passed]}"
        )
    interpretation_path = write_interpretation(artifacts_dir, output_md=output_md)
    return summary, interpretation_path


def main() -> None:
    """What: CLI wrapper for artifact validation and interpretation output.
    Why: Exposes one command for contract checks and summary generation.
    """
    # ===== CLI Args =====
    parser = argparse.ArgumentParser(description="Validate recommender artifacts.")
    parser.add_argument("--config", required=True, help="Recommender YAML config")
    args = parser.parse_args()

    # ===== Resolve Artifact Paths =====
    runtime = load_recommender_runtime_config(args.config)
    artifacts_dir = runtime.artifacts_dir
    paths = artifact_paths(runtime)

    # ===== Run =====
    summary, interpretation_path = run_validate_outputs(
        artifacts_dir,
        output_json=paths.output_validation_summary,
        output_md=paths.output_interpretation,
    )
    # ===== Write Outputs =====
    print(f"Wrote interpretation: {interpretation_path}")
    print(f"Validation passed: {paths.output_validation_summary}")


if __name__ == "__main__":
    main()
