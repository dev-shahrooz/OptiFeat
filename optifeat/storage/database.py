"""SQLite storage utilities for OptiFeat."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, List, Optional

from optifeat.config import DATABASE_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    target_column TEXT NOT NULL,
    time_budget REAL NOT NULL,
    accuracy REAL NOT NULL,
    elapsed_time REAL NOT NULL,
    selected_features TEXT NOT NULL,
    steps TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def initialize_database(path: Optional[Path] = None) -> None:
    """Ensure the SQLite database and schema exist."""
    db_path = path or DATABASE_PATH
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_connection(path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    db_path = path or DATABASE_PATH
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def record_run(
    *,
    dataset_name: str,
    target_column: str,
    time_budget: float,
    accuracy: float,
    elapsed_time: float,
    selected_features: Iterable[str],
    steps: Iterable[str],
) -> None:
    """Persist a completed run to the database."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                dataset_name,
                target_column,
                time_budget,
                accuracy,
                elapsed_time,
                selected_features,
                steps
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                target_column,
                time_budget,
                accuracy,
                elapsed_time,
                json.dumps(list(selected_features)),
                json.dumps(list(steps)),
            ),
        )
        conn.commit()


def fetch_history(*, limit: int = 50, offset: int = 0) -> List[dict]:
    """Return the most recent optimization runs."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, dataset_name, target_column, time_budget, accuracy,
                   elapsed_time, selected_features, steps, created_at
            FROM runs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row[0],
                    "dataset_name": row[1],
                    "target_column": row[2],
                    "time_budget": row[3],
                    "accuracy": row[4],
                    "elapsed_time": row[5],
                    "selected_features": json.loads(row[6]),
                    "steps": json.loads(row[7]),
                    "created_at": row[8],
                }
            )
        return results


def fetch_run(run_id: int) -> Optional[dict]:
    """Fetch a single run by its identifier."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, dataset_name, target_column, time_budget, accuracy,
                   elapsed_time, selected_features, steps, created_at
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "dataset_name": row[1],
            "target_column": row[2],
            "time_budget": row[3],
            "accuracy": row[4],
            "elapsed_time": row[5],
            "selected_features": json.loads(row[6]),
            "steps": json.loads(row[7]),
            "created_at": row[8],
        }
