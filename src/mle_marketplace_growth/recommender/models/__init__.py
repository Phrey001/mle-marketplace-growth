"""Model-family modules for the recommender engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItems:
    """What: Hold one user's ranked item indices and aligned scores.
    Why: Makes scorer top-k outputs clearer than returning a bare tuple.
    """

    item_indices: list[int]
    scores: list[float]
