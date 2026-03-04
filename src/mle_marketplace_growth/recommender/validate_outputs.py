"""Validate recommender artifacts and write a concise interpretation summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def run_validation(artifacts_dir: Path, output_json: Path | None = None) -> tuple[bool, dict]:
    # Load required artifacts.
    train_metrics_path = artifacts_dir / "train_metrics.json"
    validation_metrics_path = artifacts_dir / "validation_retrieval_metrics.json"
    test_metrics_path = artifacts_dir / "test_retrieval_metrics.json"
    index_json_path = artifacts_dir / "item_embedding_index.json"
    ann_meta_path = artifacts_dir / "ann_index_meta.json"
    topk_csv_path = artifacts_dir / "topk_recommendations.csv"
    for path in [train_metrics_path, validation_metrics_path, test_metrics_path, index_json_path, ann_meta_path, topk_csv_path]:
        if not path.exists(): raise FileNotFoundError(f"Required artifact not found: {path}")
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    validation_metrics = json.loads(validation_metrics_path.read_text(encoding="utf-8"))
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    index_json = json.loads(index_json_path.read_text(encoding="utf-8"))
    ann_meta = json.loads(ann_meta_path.read_text(encoding="utf-8"))
    with topk_csv_path.open("r", encoding="utf-8", newline="") as file:
        topk_count = len(list(csv.DictReader(file)))

    validation_rows = validation_metrics.get("rows", [])
    test_rows = test_metrics.get("rows", [])
    selected_model = train_metrics.get("selected_model_name")
    expected_models = {"popularity", "mf", "two_tower"}

    # Core consistency checks.
    checks = [
        {
            "check": "selected_model_present",
            "description": "Selected model must be one of popularity/mf/two_tower.",
            "passed": selected_model in expected_models,
            "detail": f"selected_model_name={selected_model}",
        },
        {
            "check": "baseline_rows_present",
            "description": "Validation and test metrics should contain popularity, mf, and two_tower rows.",
            "passed": {row.get("model_name") for row in validation_rows} == expected_models
            and {row.get("model_name") for row in test_rows} == expected_models,
            "detail": f"validation_rows={len(validation_rows)}, test_rows={len(test_rows)}",
        },
    ]
    # Metric bounds checks.
    metric_in_bounds = True
    for payload in [validation_rows, test_rows]:
        for row in payload:
            for key, value in row.get("metrics", {}).items():
                if key.startswith(("Recall@", "NDCG@", "HitRate@")) and not (0.0 <= float(value) <= 1.0):
                    metric_in_bounds = False
    checks.append(
        {
            "check": "metrics_in_bounds",
            "description": "Recall/NDCG/HitRate metrics should stay in [0,1].",
            "passed": metric_in_bounds,
            "detail": "all_metrics_checked",
        }
    )
    # Output completeness checks.
    checks.append(
        {
            "check": "topk_recommendations_non_empty",
            "description": "Top-K recommendation output must have at least one row.",
            "passed": topk_count > 0,
            "detail": f"row_count={topk_count}",
        }
    )
    checks.append(
        {
            "check": "embedding_index_model_matches_selection",
            "description": "Embedding index selected model should match train selected model.",
            "passed": index_json.get("selected_model_name") == selected_model,
            "detail": f"index_selected={index_json.get('selected_model_name')}",
        }
    )
    checks.append(
        {
            "check": "ann_index_artifacts_present",
            "description": "ANN metadata and FAISS ANN index artifact must be present.",
            "passed": (artifacts_dir / "ann_index.bin").exists()
            and ann_meta.get("backend") == "faiss_hnsw_ip",
            "detail": f"ann_backend={ann_meta.get('backend')}",
        }
    )

    passed = all(check["passed"] for check in checks)
    summary = {"passed": passed, "artifacts_dir": str(artifacts_dir), "checks": checks}
    # Optionally persist validation summary.
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return passed, summary


def write_interpretation(artifacts_dir: Path, output_md: Path | None = None) -> Path:
    # Load summary artifacts.
    train_metrics = json.loads((artifacts_dir / "train_metrics.json").read_text(encoding="utf-8"))
    validation_metrics = json.loads((artifacts_dir / "validation_retrieval_metrics.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((artifacts_dir / "test_retrieval_metrics.json").read_text(encoding="utf-8"))
    selected_model = train_metrics.get("selected_model_name", "unknown")
    selection_rule = train_metrics.get("selection_rule", "")
    catalog_size = int(train_metrics.get("counts", {}).get("items_train_universe", 0))
    k_values = [int(value) for value in train_metrics.get("k_values", []) if int(value) > 0]
    anchor_k = 20 if 20 in k_values else (max(k_values) if k_values else 20)
    random_anchor = (anchor_k / catalog_size) if catalog_size > 0 else 0.0

    val_by_model = {row["model_name"]: row.get("metrics", {}) for row in validation_metrics.get("rows", [])}
    test_by_model = {row["model_name"]: row.get("metrics", {}) for row in test_metrics.get("rows", [])}
    def _lift(observed: float) -> str:
        if random_anchor <= 0:
            return "n/a"
        return f"{(observed / random_anchor):.2f}x"

    # Build compact interpretation report.
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
    report_path = output_md or (artifacts_dir / "output_interpretation.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    # Parse CLI arguments.
    parser = argparse.ArgumentParser(description="Validate recommender artifacts.")
    parser.add_argument("--artifacts-dir", default="artifacts/recommender", help="Recommender artifacts directory")
    parser.add_argument("--output-json", default="artifacts/recommender/output_validation_summary.json", help="Path for validation summary output")
    args = parser.parse_args()

    # Run checks and emit interpretation.
    artifacts_dir = Path(args.artifacts_dir)
    passed, summary = run_validation(artifacts_dir, output_json=Path(args.output_json))
    if not passed: raise SystemExit(f"Validation failed: {[row for row in summary['checks'] if not row['passed']]}")
    interpretation_path = write_interpretation(artifacts_dir)
    print(f"Wrote interpretation: {interpretation_path}")
    print(f"Validation passed: {args.output_json}")


if __name__ == "__main__":
    main()
