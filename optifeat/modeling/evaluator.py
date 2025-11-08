"""Model evaluation helpers for feature selection."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class FeatureEvaluation:
    """Stores the evaluation metrics for a single feature or feature set."""

    features: Sequence[str]
    accuracy: float
    elapsed_time: float


class ModelEvaluator:
    """Evaluates feature subsets using a simple classification model."""

    def __init__(self, *, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def _prepare_data(
        self, frame: pd.DataFrame, *, target_column: str, features: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = frame[list(features)].values
        y = frame[target_column].values
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

    def evaluate_features(
        self, frame: pd.DataFrame, *, target_column: str, features: Iterable[str]
    ) -> FeatureEvaluation:
        features = list(features)
        if not features:
            raise ValueError("At least one feature must be provided for evaluation.")

        X_train, X_test, y_train, y_test = self._prepare_data(
            frame, target_column=target_column, features=features
        )

        model = LogisticRegression(max_iter=500)
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        elapsed = time.perf_counter() - start_time

        accuracy = accuracy_score(y_test, predictions)
        return FeatureEvaluation(features=features, accuracy=float(accuracy), elapsed_time=elapsed)
