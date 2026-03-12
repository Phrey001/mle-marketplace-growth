"""Shared constants for recommender training, serving, and validation flows.

Usage Steps:
1) Keep immutable allowed-value sets in one place.
2) Keep deterministic defaults used by config-first runtime.
3) Keep ANN backend parameters synchronized across predict/validate.
"""

MODEL_NAMES = ("popularity", "mf", "two_tower")
EXPECTED_MODELS = set(MODEL_NAMES)

DEFAULT_ARTIFACTS_DIR = "artifacts/recommender"
DEFAULT_TOP_K = 20
DEFAULT_TOP_KS = "10,20"

ALLOWED_MF_WEIGHTINGS = ("binary", "tfidf")
DEFAULT_MF_WEIGHTING = "tfidf"

EARLY_STOP_METRIC = "val_recall_at_k"
NORMALIZE_EMBEDDINGS = True
DEVICE = "auto"
MF_ALGORITHM = "randomized"
POPULARITY_TRANSFORM = "log1p"

ANN_BACKEND = "faiss_hnsw_ip"
ANN_METRIC = "inner_product"
ANN_HNSW_M = 32
ANN_HNSW_EF_CONSTRUCTION = 80
