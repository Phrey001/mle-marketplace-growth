from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_ann_index(output_dir: Path, item_embeddings: np.ndarray) -> dict:
    ann_index_path = output_dir / "ann_index.bin"
    embeddings = np.asarray(item_embeddings, dtype=np.float32)
    dim = int(embeddings.shape[1])
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.add(embeddings)
    faiss.write_index(index, str(ann_index_path))
    return {
        "backend": "faiss_hnsw_ip",
        "metric": "inner_product",
        "dimension": dim,
        "item_count": int(embeddings.shape[0]),
        "notes": "ANN retrieval enabled via FAISS HNSW index.",
    }
