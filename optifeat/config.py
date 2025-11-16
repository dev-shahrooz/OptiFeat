"""Application configuration settings."""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Data handling
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_PATH = PROJECT_ROOT / "optifeat.db"

# Modeling defaults
DEFAULT_TARGET_COLUMN = "target"
DEFAULT_TIME_BUDGET = 5.0  # seconds
DEFAULT_MIN_COST_MS = 1  # milliseconds
DEFAULT_SCALE_FACTOR_MS = 1  # milliseconds per DP unit

# Web application configuration
SECRET_KEY = "optifeat-secret-key"

# Pagination and dashboard
HISTORY_PAGE_SIZE = 10
