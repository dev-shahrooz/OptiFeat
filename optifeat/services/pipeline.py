"""End-to-end pipeline orchestrating dataset loading, evaluation, and optimization."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from optifeat.config import (
    DEFAULT_MIN_COST_MS,
    DEFAULT_SCALE_FACTOR_MS,
    DEFAULT_TIME_BUDGET,
)
from optifeat.data.dataset_loader import DatasetLoader
from optifeat.modeling.evaluator import FeatureEvaluation, ModelEvaluator
from optifeat.optimization.knapsack_selector import FeatureCandidate, KnapsackSelector
from optifeat.storage import database


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run."""

    metadata: dict
    best_evaluation: FeatureEvaluation
    selected_features: List[str]
    candidate_summary: List[dict]
    total_candidate_cost_ms: int
    selected_cost_ms: int
    steps: List[str] = field(default_factory=list)

    @property
    def total_candidate_cost_seconds(self) -> float:
        return self.total_candidate_cost_ms / 1000.0

    @property
    def selected_cost_seconds(self) -> float:
        return self.selected_cost_ms / 1000.0


class OptimizationPipeline:
    """Coordinates the end-to-end feature selection workflow."""

    def __init__(self) -> None:
        self.loader = DatasetLoader()
        self.evaluator = ModelEvaluator()

    def _log(self, steps: List[str], message: str) -> None:
        steps.append(message)

    def _prepare_frame(self, frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
        processed = frame.copy()
        for column in processed.columns:
            series = processed[column]
            if series.isna().any():
                if pd.api.types.is_numeric_dtype(series):
                    fill_value = series.median()
                    if pd.isna(fill_value):
                        fill_value = 0
                    processed[column] = series.fillna(fill_value)
                else:
                    mode = series.mode()
                    fill_value = mode.iloc[0] if not mode.empty else "missing"
                    processed[column] = series.fillna(fill_value)

            if column == target_column:
                continue

            if not pd.api.types.is_numeric_dtype(processed[column]):
                processed[column], _ = pd.factorize(processed[column])

        if processed[target_column].isna().any():
            processed = processed.dropna(subset=[target_column])

        target_series = processed[target_column]
        if not pd.api.types.is_numeric_dtype(target_series):
            processed[target_column], _ = pd.factorize(target_series)

        return processed

    def _evaluate_individual_features(
        self, frame: pd.DataFrame, *, target_column: str
    ) -> Dict[str, FeatureEvaluation]:
        evaluations: Dict[str, FeatureEvaluation] = {}
        for column in frame.columns:
            if column == target_column:
                continue
            evaluation = self.evaluator.evaluate_features(
                frame, target_column=target_column, features=[column]
            )
            evaluations[column] = evaluation
        return evaluations

    def _create_candidates(
        self,
        evaluations: Dict[str, FeatureEvaluation],
        *,
        min_cost_ms: int,
    ) -> List[FeatureCandidate]:
        candidates: List[FeatureCandidate] = []
        for name, evaluation in evaluations.items():
            elapsed_ms = int(evaluation.elapsed_time * 1000)
            cost_ms = max(elapsed_ms, min_cost_ms)
            candidates.append(
                FeatureCandidate(name=name, value=evaluation.accuracy, cost_ms=cost_ms)
            )
        return candidates

    def run(
        self,
        dataset_path: Path,
        *,
        target_column: str,
        time_budget: float = DEFAULT_TIME_BUDGET,
        min_cost_ms: int = DEFAULT_MIN_COST_MS,
        scale_factor_ms: int = DEFAULT_SCALE_FACTOR_MS,
    ) -> PipelineResult:
        steps: List[str] = []
        self._log(steps, "Loading dataset ...")
        frame, metadata = self.loader.load(dataset_path)

        if target_column not in frame.columns:
            raise ValueError(
                f"Target column '{target_column}' is not present in the dataset."
            )

        processed_frame = self._prepare_frame(frame, target_column)
        if processed_frame[target_column].nunique() < 2:
            raise ValueError("ستون هدف باید حداقل شامل دو کلاس باشد.")

        self._log(steps, "Evaluating individual features ...")
        evaluations = self._evaluate_individual_features(
            processed_frame, target_column=target_column
        )
        self._log(steps, f"{len(evaluations)} ویژگی تحلیل شد.")

        self._log(steps, "Preparing knapsack candidates ...")
        min_cost_ms = max(1, int(min_cost_ms))
        scale_factor_ms = max(1, int(scale_factor_ms))
        candidates = self._create_candidates(
            evaluations, min_cost_ms=min_cost_ms
        )
        selector = KnapsackSelector(
            time_budget=time_budget, time_unit_ms=scale_factor_ms
        )

        self._log(steps, "Running knapsack optimization ...")
        result = selector.select(candidates)
        self._log(
            steps,
            f"{len(result.selected_features)} ویژگی در محدودیت زمانی {time_budget} ثانیه انتخاب شد.",
        )

        selected_features = result.selected_features
        if not selected_features:
            raise RuntimeError(
                "No features were selected within the provided time budget."
            )

        self._log(steps, "Evaluating selected feature subset ...")
        best_evaluation = self.evaluator.evaluate_features(
            processed_frame, target_column=target_column, features=selected_features
        )
        self._log(
            steps,
            f"دقت نهایی مجموعه ویژگی‌ها: {best_evaluation.accuracy:.3f}",
        )

        metadata_dict = {
            "dataset_name": metadata.path.name,
            "num_rows": metadata.num_rows,
            "num_features": metadata.num_features,
            "target_column": target_column,
            "time_budget": time_budget,
            "time_budget_ms": int(time_budget * 1000),
            "min_cost_ms": min_cost_ms,
            "scale_factor_ms": scale_factor_ms,
        }

        candidate_frame = describe_candidates(candidates)
        total_candidate_cost_ms = (
            int(candidate_frame["cost_ms"].sum()) if not candidate_frame.empty else 0
        )
        selected_cost_ms = result.total_cost_ms

        database.record_run(
            dataset_name=metadata.path.name,
            target_column=target_column,
            time_budget=time_budget,
            accuracy=best_evaluation.accuracy,
            elapsed_time=best_evaluation.elapsed_time,
            selected_features=selected_features,
            steps=steps,
            total_cost_all_features=total_candidate_cost_ms / 1000.0,
            selected_cost=selected_cost_ms / 1000.0,
        )

        return PipelineResult(
            metadata=metadata_dict,
            best_evaluation=best_evaluation,
            selected_features=selected_features,
            candidate_summary=candidate_frame.to_dict(orient="records"),
            steps=steps,
            total_candidate_cost_ms=total_candidate_cost_ms,
            selected_cost_ms=selected_cost_ms,
        )


def describe_candidates(candidates: Iterable[FeatureCandidate]) -> pd.DataFrame:
    """Return a dataframe summarizing candidate statistics."""
    data = [
        {
            "feature": c.name,
            "accuracy": c.value,
            "cost_ms": c.cost_ms,
            "cost_seconds": c.cost_ms / 1000.0,
        }
        for c in candidates
    ]
    if not data:
        return pd.DataFrame(
            columns=["feature", "accuracy", "cost_ms", "cost_seconds"]
        )
    return pd.DataFrame(data)
