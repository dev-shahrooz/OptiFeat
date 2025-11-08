"""Utilities for loading and validating datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetMetadata:
    """Metadata describing an uploaded dataset."""

    path: Path
    target_column: str
    num_rows: int
    num_features: int


class DatasetLoader:
    """Handles reading tabular datasets into pandas DataFrames."""

    def __init__(self, *, target_column: Optional[str] = None) -> None:
        self.target_column = target_column

    def load(self, file_path: Path) -> tuple[pd.DataFrame, DatasetMetadata]:
        """Load a CSV dataset and validate the target column."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        frame = pd.read_csv(file_path)
        if frame.empty:
            raise ValueError("The provided dataset is empty.")

        target_column = self.target_column or frame.columns[-1]
        if target_column not in frame.columns:
            raise ValueError(
                f"Target column '{target_column}' is not present in the dataset."
            )

        metadata = DatasetMetadata(
            path=file_path,
            target_column=target_column,
            num_rows=len(frame),
            num_features=len(frame.columns) - 1,
        )

        return frame, metadata
