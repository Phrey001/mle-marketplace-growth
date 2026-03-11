"""Train recommender baselines and a two-tower-style retrieval model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from mle_marketplace_growth.recommender.artifacts import _write_ann_index, _write_json
from mle_marketplace_growth.recommender.data import _build_interactions, _load_entity_index, _load_split_rows, _validate_split_chronology
from mle_marketplace_growth.recommender.eval import _evaluate_model, _user_eval_pool  # re-exported for existing tests/imports
from mle_marketplace_growth.recommender.models import _popularity_scores, _train_mf, _train_two_tower

EARLY_STOP_METRIC = "val_recall_at_k"
NORMALIZE_EMBEDDINGS = True
DEVICE = "auto"
MF_ALGORITHM = "randomized"
POPULARITY_TRANSFORM = "log1p"


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Train recommender retrieval models.")
    parser.add_argument("--splits-path", default="data/gold/feature_store/recommender/user_item_splits/user_item_splits.parquet", help="Path to user_item_splits parquet")
    parser.add_argument("--user-index-path", default="data/gold/feature_store/recommender/user_index/user_index.parquet", help="Path to user_index parquet")
    parser.add_argument("--item-index-path", default="data/gold/feature_store/recommender/item_index/item_index.parquet", help="Path to item_index parquet")
    parser.add_argument("--output-dir", default="artifacts/recommender", help="Output artifact directory")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension for two-tower model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for two-tower model")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for two-tower model")
    parser.add_argument("--negative-samples", type=int, default=3, help="Negative samples per positive pair")
    parser.add_argument("--batch-size", type=int, default=4096, help="Two-tower positive-pair batch size")
    parser.add_argument("--l2-reg", type=float, default=1e-4, help="L2 regularization strength for two-tower updates")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm for two-tower training (0 disables)")
    parser.add_argument("--early-stop-rounds", type=int, default=3, help="Stop two-tower after this many non-improving epochs (0 disables)")
    parser.add_argument("--early-stop-k", type=int, default=20, help="K used by validation Recall@K early stopping")
    parser.add_argument("--early-stop-tolerance", type=float, default=1e-4, help="Minimum metric improvement to count as progress")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for two-tower logits (>0)")
    parser.add_argument("--tower-hidden-dim", type=int, default=0, help="Two-tower MLP hidden dimension (0 disables MLP tower)")
    parser.add_argument("--tower-dropout", type=float, default=0.0, help="Two-tower MLP dropout rate")
    parser.add_argument("--mf-components", type=int, default=32, help="Latent factors for MF baseline")
    parser.add_argument("--mf-n-iter", type=int, default=15, help="Iteration budget for MF SVD solver")
    parser.add_argument("--mf-weighting", choices=["binary", "tfidf"], default="tfidf", help="Input weighting mode for MF")
    parser.add_argument("--top-ks", default="10,20", help="Comma-separated K values for offline metrics")
    args = parser.parse_args()

    # ===== Input Checks =====
    split_path, output_dir = Path(args.splits_path), Path(args.output_dir)
    if not split_path.exists(): raise FileNotFoundError(f"Split CSV not found: {split_path}")
    top_ks = sorted({int(value.strip()) for value in args.top_ks.split(",") if value.strip()})
    if not top_ks: raise ValueError("At least one K is required in --top-ks")
    if args.embedding_dim < 2: raise ValueError("--embedding-dim must be >= 2")

    # ===== Load Inputs =====
    rows = _load_split_rows(split_path)
    _validate_split_chronology(rows)
    train, validation, test = _build_interactions(rows)
    user_ids, user_to_idx = _load_entity_index(Path(args.user_index_path), id_col="user_id", idx_col="user_idx")
    item_ids, item_to_idx = _load_entity_index(Path(args.item_index_path), id_col="item_id", idx_col="item_idx")
    print(
        "[recommender.train] loaded splits:",
        f"train_users={len(train)}, val_users={len(validation)}, test_users={len(test)}, item_universe={len(item_ids)}",
    )
    print(
        "[recommender.train] config:",
        f"embedding_dim={args.embedding_dim}, epochs={args.epochs}, lr={args.learning_rate}, negatives={args.negative_samples}, batch_size={args.batch_size}, l2={args.l2_reg}, max_grad_norm={args.max_grad_norm}",
    )
    print(
        "[recommender.train] convergence:",
        f"early_stop_rounds={args.early_stop_rounds}, early_stop_metric={EARLY_STOP_METRIC}, early_stop_tolerance={args.early_stop_tolerance}, early_stop_k={args.early_stop_k}, temperature={args.temperature}, normalize_embeddings={NORMALIZE_EMBEDDINGS}, tower_hidden_dim={args.tower_hidden_dim}, tower_dropout={args.tower_dropout}, device={DEVICE}, mf_algorithm={MF_ALGORITHM}",
    )

    # ===== Train Models =====
    popularity = _popularity_scores(train, item_to_idx, transform=POPULARITY_TRANSFORM)
    print("[recommender.train] trained popularity baseline")
    mf_user, mf_item = _train_mf(
        train,
        user_to_idx,
        item_to_idx,
        args.mf_components,
        n_iter=args.mf_n_iter,
        weighting=args.mf_weighting,
        algorithm=MF_ALGORITHM,
        tol=0.0,
    )
    print(
        f"[recommender.train] trained mf baseline (components={args.mf_components}, n_iter={args.mf_n_iter}, weighting={args.mf_weighting}, algorithm={MF_ALGORITHM})"
    )
    tt_user, tt_item = _train_two_tower(
        train,
        user_to_idx,
        item_to_idx,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        negative_samples=args.negative_samples,
        batch_size=max(1, args.batch_size),
        l2_reg=args.l2_reg,
        max_grad_norm=max(0.0, args.max_grad_norm),
        early_stop_rounds=max(0, args.early_stop_rounds),
        early_stop_metric=EARLY_STOP_METRIC,
        early_stop_k=max(1, args.early_stop_k),
        early_stop_tolerance=max(0.0, args.early_stop_tolerance),
        validation_interactions=validation,
        temperature=float(args.temperature),
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        tower_hidden_dim=max(0, args.tower_hidden_dim),
        tower_dropout=max(0.0, args.tower_dropout),
        device=DEVICE,
        verbose=True,
    )
    if NORMALIZE_EMBEDDINGS:
        tt_user, tt_item = _l2_normalize_rows(tt_user), _l2_normalize_rows(tt_item)
    print("[recommender.train] trained two_tower")

    # ===== Evaluate Candidates =====
    model_names = ["popularity", "mf", "two_tower"]
    metrics_by_split = {
        split_name: [
            _evaluate_model(
                model_name=name,
                users=user_ids,
                train=train,
                split_rows=split_rows,
                user_to_idx=user_to_idx,
                item_to_idx=item_to_idx,
                top_ks=top_ks,
                popularity=popularity,
                mf_user=mf_user,
                mf_item=mf_item,
                tt_user=tt_user,
                tt_item=tt_item,
            )
            for name in model_names
        ]
        for split_name, split_rows in [("validation", validation), ("test", test)]
    }
    validation_metrics, test_metrics = metrics_by_split["validation"], metrics_by_split["test"]
    select_k = 20 if 20 in top_ks else max(top_ks)
    for row in validation_metrics:
        print(
            "[recommender.train] validation:",
            row["model_name"],
            f"Recall@{select_k}={row['metrics'].get(f'Recall@{select_k}', 0.0):.6f}",
        )

    selected_model_name = max(validation_metrics, key=lambda row: row["metrics"].get(f"Recall@{select_k}", 0.0))["model_name"]
    print(f"[recommender.train] selected_model={selected_model_name} by Recall@{select_k}")
    selected_test_row = next((row for row in test_metrics if row["model_name"] == selected_model_name), None)
    if selected_test_row is not None:
        print(
            "[recommender.train] test:",
            selected_model_name,
            f"Recall@{select_k}={selected_test_row['metrics'].get(f'Recall@{select_k}', 0.0):.6f}",
        )
    selected_item_embeddings = {"two_tower": tt_item, "mf": mf_item, "popularity": popularity.reshape(-1, 1)}[selected_model_name]

    # ===== Write Outputs =====
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "item_embeddings.npy", selected_item_embeddings)
    _write_json(output_dir / "item_embedding_index.json", {
        "selected_model_name": selected_model_name,
        "item_to_row_index": item_to_idx,
        "embedding_shape": list(selected_item_embeddings.shape),
    })
    ann_metadata = _write_ann_index(output_dir, selected_item_embeddings)
    _write_json(output_dir / "ann_index_meta.json", ann_metadata)

    with (output_dir / "model_bundle.pkl").open("wb") as file:
        pickle.dump(
            {
                "selected_model_name": selected_model_name,
                "user_ids": user_ids,
                "item_ids": item_ids,
                "user_to_idx": user_to_idx,
                "item_to_idx": item_to_idx,
                "train_user_items": train,
                "top_ks": top_ks,
                "popularity_scores": popularity,
                "mf_user_embeddings": mf_user,
                "mf_item_embeddings": mf_item,
                "two_tower_user_embeddings": tt_user,
                "two_tower_item_embeddings": tt_item,
            },
            file,
        )

    _write_json(output_dir / "validation_retrieval_metrics.json", {"rows": validation_metrics, "k_values": top_ks})
    _write_json(output_dir / "test_retrieval_metrics.json", {"rows": test_metrics, "k_values": top_ks})
    _write_json(output_dir / "train_metrics.json", {
        "input_splits_path": str(split_path),
        "selected_model_name": selected_model_name,
        "selection_rule": f"maximize_validation_Recall@{select_k}",
        "k_values": top_ks,
        "counts": {
            "users_total": len(user_ids),
            "items_train_universe": len(item_ids),
            "train_users": len(train),
            "validation_users": len(validation),
            "test_users": len(test),
        },
        "model_config": {
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "negative_samples": args.negative_samples,
            "batch_size": max(1, args.batch_size),
            "l2_reg": args.l2_reg,
            "max_grad_norm": max(0.0, args.max_grad_norm),
            "early_stop_rounds": max(0, args.early_stop_rounds),
            "early_stop_metric": EARLY_STOP_METRIC,
            "early_stop_k": max(1, args.early_stop_k),
            "early_stop_tolerance": max(0.0, args.early_stop_tolerance),
            "temperature": float(args.temperature),
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "tower_hidden_dim": max(0, args.tower_hidden_dim),
            "tower_dropout": max(0.0, args.tower_dropout),
            "device": DEVICE,
            "mf_components": args.mf_components,
            "mf_n_iter": args.mf_n_iter,
            "mf_weighting": args.mf_weighting,
            "mf_algorithm": MF_ALGORITHM,
            "mf_tol": 0.0,
            "popularity_transform": POPULARITY_TRANSFORM,
            "ann_backend": ann_metadata.get("backend"),
        },
    })
    print(f"Wrote recommender train artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
