from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _load_split_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows: raise ValueError(f"No rows found in split CSV: {path}")
    required = {"user_id", "item_id", "split", "event_ts"}
    missing = sorted(required - set(rows[0].keys()))
    if missing: raise ValueError(f"Missing required columns in split CSV: {missing}")
    return rows


def _validate_split_chronology(rows: list[dict]) -> None:
    per_user = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for row in rows:
        split = row["split"].strip().lower()
        if split == "validation":
            split = "val"
        if split not in {"train", "val", "test"}:
            continue
        per_user[row["user_id"]][split].append(datetime.fromisoformat(row["event_ts"]))

    violations = 0
    for buckets in per_user.values():
        train_ts = buckets["train"]
        val_ts = buckets["val"]
        test_ts = buckets["test"]
        if not train_ts or not val_ts or not test_ts:
            continue
        if max(train_ts) > min(val_ts):
            violations += 1
        if max(train_ts) > min(test_ts):
            violations += 1
        if min(val_ts) > min(test_ts):
            violations += 1
    if violations > 0: raise ValueError(f"Split chronology violation detected: {violations} user-level ordering violations.")


def _build_interactions(rows: list[dict]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    train = defaultdict(set)
    validation = defaultdict(set)
    test = defaultdict(set)
    for row in rows:
        split = row["split"].strip().lower()
        user_id = row["user_id"]
        item_id = row["item_id"]
        if split == "train":
            train[user_id].add(item_id)
        elif split in {"validation", "val"}:
            validation[user_id].add(item_id)
        elif split == "test":
            test[user_id].add(item_id)
    if not train: raise ValueError("No train interactions found.")
    if not validation: raise ValueError("No validation interactions found.")
    if not test: raise ValueError("No test interactions found.")
    return dict(train), dict(validation), dict(test)


def _load_entity_index(path: Path, id_col: str, idx_col: str) -> tuple[list[str], dict[str, int]]:
    if not path.exists(): raise FileNotFoundError(f"Entity index CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows: raise ValueError(f"No rows found in entity index CSV: {path}")
    required = {id_col, idx_col}
    missing = sorted(required - set(rows[0].keys()))
    if missing: raise ValueError(f"Missing required columns in entity index CSV {path}: {missing}")

    idx_to_id: dict[int, str] = {}
    id_to_idx: dict[str, int] = {}
    for row in rows:
        entity_id = row[id_col]
        entity_idx = int(row[idx_col])
        if entity_id in id_to_idx: raise ValueError(f"Duplicate entity id in index CSV {path}: {entity_id}")
        if entity_idx in idx_to_id: raise ValueError(f"Duplicate entity idx in index CSV {path}: {entity_idx}")
        id_to_idx[entity_id] = entity_idx
        idx_to_id[entity_idx] = entity_id

    expected = list(range(len(rows)))
    observed = sorted(idx_to_id.keys())
    if observed != expected: raise ValueError(f"Entity idx must be contiguous from 0 in {path}")
    return [idx_to_id[idx] for idx in expected], id_to_idx
