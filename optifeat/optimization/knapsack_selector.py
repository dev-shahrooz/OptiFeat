"""Knapsack-based feature selector."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class FeatureCandidate:
    """Represents a feature with value (accuracy) and cost (milliseconds)."""

    name: str
    value: float
    cost_ms: int


@dataclass
class KnapsackResult:
    """Result of the knapsack optimization."""

    selected_features: List[str]
    total_value: float
    total_cost_ms: int

    @property
    def total_cost_seconds(self) -> float:
        return self.total_cost_ms / 1000.0

class KnapsackSelector:
    """Solve the 0/1 knapsack problem for feature selection."""

    def __init__(self, time_budget: float, *, time_unit_ms: int = 1) -> None:
        self.time_budget = time_budget
        self.time_unit_ms = max(1, int(time_unit_ms))


    def select(self, candidates: Sequence[FeatureCandidate]) -> KnapsackResult:
        if not candidates:
            return KnapsackResult(
                selected_features=[], total_value=0.0, total_cost_ms=0
            )

        capacity_units = max(
            int(self.time_budget * 1000) // self.time_unit_ms,
            0,
        )
        if capacity_units == 0:
            return KnapsackResult(
                selected_features=[],
                total_value=0.0,
                total_cost_ms=0,
            )

        weights = [
            max(1, math.ceil(candidate.cost_ms / self.time_unit_ms))
            for candidate in candidates
        ]
        values = [c.value for c in candidates]

        n = len(candidates)
        dp = [[0.0 for _ in range(capacity_units + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            weight = weights[i - 1]
            value = values[i - 1]
            for w in range(capacity_units  + 1):
                dp[i][w] = dp[i - 1][w]
                if weight <= w:
                    candidate_value = dp[i - 1][w - weight] + value
                    if candidate_value > dp[i][w]:
                        dp[i][w] = candidate_value

        selected: List[str] = []
        total_cost_ms = 0
        w = capacity_units
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                candidate = candidates[i - 1]
                selected.append(candidate.name)
                total_cost_ms += candidate.cost_ms
                w -= weights[i - 1]

        selected.reverse()
        total_value = dp[n][capacity_units]
        return KnapsackResult(
            selected_features=selected,
            total_value=total_value,
            total_cost_ms=total_cost_ms,
        )
