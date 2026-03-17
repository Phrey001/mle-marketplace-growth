"""Shared constants for recommender training, serving, and validation flows.

Usage Steps:
1) Keep immutable allowed-value sets in one place.
2) Keep implementation policy synchronized across train/predict/validate.
3) Keep ANN backend parameters synchronized across predict/validate.
"""

MODEL_NAMES = ("popularity", "mf", "two_tower")  # What: supported recommender model families. Why: keeps train/eval/validate model ordering aligned.
EXPECTED_MODELS = set(MODEL_NAMES)  # What: set form of supported model names. Why: validation checks use membership/coverage logic.

EVALUATION_TOP_K = 20  # What: fixed evaluation cutoff. Why: recommender selection/reporting is standardized on Recall@20.
ALLOWED_MF_WEIGHTINGS = ("binary", "tfidf")  # What: supported MF preprocessing modes. Why: fail fast on unsupported weighting labels.

EARLY_STOP_METRIC = "val_recall_at_k"  # What: early-stop signal name. Why: train_and_select.py and the model helper should log/store the same policy label.
NORMALIZE_EMBEDDINGS = True  # What: L2-normalize user/item vectors. Why: two-tower scoring uses cosine-style dot product.
DEVICE = "auto"  # What: runtime device policy. Why: repo chooses CUDA when available, otherwise CPU, without user-facing override.
MF_ALGORITHM = "randomized"  # What: TruncatedSVD solver mode. Why: stable fast default for this demo-scale MF baseline.
POPULARITY_TRANSFORM = "log1p"  # What: popularity-score transform. Why: dampens head-item dominance before ranking.

ANN_BACKEND = "faiss_hnsw_ip"  # What: ANN backend contract label. Why: predict/validate must agree on the saved ANN artifact type.
ANN_METRIC = "inner_product"  # What: ANN similarity metric label. Why: serving metadata should state the retrieval scoring convention.
ANN_HNSW_M = 32  # What: HNSW graph connectivity. Why: keeps ANN build settings fixed and reproducible across runs.
ANN_HNSW_EF_CONSTRUCTION = 80  # What: HNSW construction breadth. Why: fixes ANN build quality/speed tradeoff for this repo.
