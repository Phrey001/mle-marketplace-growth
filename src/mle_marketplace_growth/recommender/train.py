"""Train recommender baselines and a two-tower-style retrieval model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from mle_marketplace_growth.recommender.artifacts import _write_ann_index, _write_json
from mle_marketplace_growth.recommender.data import _build_interactions, _load_entity_index, _load_split_rows, _validate_split_chronology
from mle_marketplace_growth.recommender.eval import _evaluate_model, _user_eval_pool
from mle_marketplace_growth.recommender.models import _popularity_scores, _train_mf, _train_two_tower


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recommender retrieval models.")
    parser.add_argument("--splits-csv", default="data/gold/feature_store/recommender/user_item_splits/user_item_splits.csv", help="Path to user_item_splits.csv")
    parser.add_argument("--user-index-csv", default="data/gold/feature_store/recommender/user_index/user_index.csv", help="Path to user_index.csv")
    parser.add_argument("--item-index-csv", default="data/gold/feature_store/recommender/item_index/item_index.csv", help="Path to item_index.csv")
    parser.add_argument("--output-dir", default="artifacts/recommender", help="Output artifact directory")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension for two-tower model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for two-tower model")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for two-tower model")
    parser.add_argument("--negative-samples", type=int, default=3, help="Negative samples per positive pair")
    parser.add_argument("--l2-reg", type=float, default=1e-4, help="L2 regularization strength for two-tower updates")
    parser.add_argument("--mf-components", type=int, default=32, help="Latent factors for MF baseline")
    parser.add_argument("--top-ks", default="10,20", help="Comma-separated K values for offline metrics")
    args = parser.parse_args()

    split_path, output_dir = Path(args.splits_csv), Path(args.output_dir)
    if not split_path.exists(): raise FileNotFoundError(f"Split CSV not found: {split_path}")
    top_ks = sorted({int(value.strip()) for value in args.top_ks.split(",") if value.strip()})
    if not top_ks: raise ValueError("At least one K is required in --top-ks")
    if args.embedding_dim < 2: raise ValueError("--embedding-dim must be >= 2")

    rows = _load_split_rows(split_path)
    _validate_split_chronology(rows)
    train, validation, test = _build_interactions(rows)
    user_ids, user_to_idx = _load_entity_index(Path(args.user_index_csv), id_col="user_id", idx_col="user_idx")
    item_ids, item_to_idx = _load_entity_index(Path(args.item_index_csv), id_col="item_id", idx_col="item_idx")

    popularity = _popularity_scores(train, item_to_idx)
    mf_user, mf_item = _train_mf(train, user_to_idx, item_to_idx, args.mf_components)
    tt_user, tt_item = _train_two_tower(
        train,
        user_to_idx,
        item_to_idx,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        negative_samples=args.negative_samples,
        l2_reg=args.l2_reg,
    )

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
    selected_model_name = max(validation_metrics, key=lambda row: row["metrics"].get(f"Recall@{select_k}", 0.0))["model_name"]
    selected_item_embeddings = {"two_tower": tt_item, "mf": mf_item, "popularity": popularity.reshape(-1, 1)}[selected_model_name]

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
        "input_splits_csv": str(split_path),
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
            "l2_reg": args.l2_reg,
            "mf_components": args.mf_components,
            "ann_backend": ann_metadata.get("backend"),
        },
    })
    print(f"Wrote recommender train artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
