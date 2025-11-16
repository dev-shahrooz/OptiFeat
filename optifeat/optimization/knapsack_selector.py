"""Knapsack-based feature selector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class FeatureCandidate:
    """Represents a feature with value (accuracy) and cost (time)."""

    name: str
    value: float
    cost: float


@dataclass
class KnapsackResult:
    """Result of the knapsack optimization."""

    selected_features: List[str]
    total_value: float
    total_cost: float


class KnapsackSelector:
    """Solve the 0/1 knapsack problem for feature selection."""

    def __init__(self, time_budget: float) -> None:
        self.time_budget = time_budget

    def select(self, candidates: Sequence[FeatureCandidate]) -> KnapsackResult:
        if not candidates:
            return KnapsackResult(selected_features=[], total_value=0.0, total_cost=0.0)

        # Discretize the budget to allow dynamic programming on floats.
        scale_factor = 10000
        capacity = int(self.time_budget * scale_factor)
        weights = [int(c.cost * scale_factor) for c in candidates]
        values = [c.value for c in candidates]

        n = len(candidates)
        dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            weight = weights[i - 1]
            value = values[i - 1]
            for w in range(capacity + 1):
                dp[i][w] = dp[i - 1][w]
                if weight <= w:
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - weight] + value)

        selected: List[str] = []
        total_cost = 0.0
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                candidate = candidates[i - 1]
                selected.append(candidate.name)
                total_cost += candidate.cost
                w -= weights[i - 1]

        selected.reverse()
        total_value = sum(c.value for c in candidates if c.name in selected)
        return KnapsackResult(
            selected_features=selected,
            total_value=total_value,
            total_cost=total_cost,
        )
