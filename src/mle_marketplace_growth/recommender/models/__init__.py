"""Model-family modules for the recommender engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItems:
    item_indices: list[int]
    scores: list[float]
