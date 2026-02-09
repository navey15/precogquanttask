"""Central configuration for the quant research pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

ENV_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ENV_ROOT.parent
DATA_DIR = WORKSPACE_ROOT / "anonymized_data"
ARTIFACTS_DIR = ENV_ROOT / "artifacts"
FEATURE_STORE = ARTIFACTS_DIR / "features" / "feature_matrix.parquet"
# Backward-compatible alias for older references
FEATURE_STORE_PATH = FEATURE_STORE
ALPHA_STORE = ARTIFACTS_DIR / "features" / "alpha_signals.parquet"
ALPHA_STORE_PATH = ALPHA_STORE
ALPHA_STORE_SMOOTHED = ARTIFACTS_DIR / "features" / "alpha_signals_smoothed.parquet"
MODEL_DIR = ARTIFACTS_DIR / "models"
PLOT_DIR = ARTIFACTS_DIR / "plots"
REPORT_DIR = ARTIFACTS_DIR / "reports"

DEFAULT_COST_BPS = 10  # 10 basis points per side
ANNUAL_TRADING_DAYS = 252
RISK_FREE_RATE = 0.0


def ensure_directories() -> Dict[str, Path]:
    """Ensure that every artifact directory exists before use."""
    paths = {
        "artifacts": ARTIFACTS_DIR,
        "feature_store": FEATURE_STORE.parent,
        "models": MODEL_DIR,
        "plots": PLOT_DIR,
        "reports": REPORT_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = [
    "ENV_ROOT",
    "WORKSPACE_ROOT",
    "DATA_DIR",
    "ARTIFACTS_DIR",
    "FEATURE_STORE",
    "FEATURE_STORE_PATH",
    "ALPHA_STORE",
    "ALPHA_STORE_PATH",
    "ALPHA_STORE_SMOOTHED",
    "MODEL_DIR",
    "PLOT_DIR",
    "REPORT_DIR",
    "DEFAULT_COST_BPS",
    "ANNUAL_TRADING_DAYS",
    "RISK_FREE_RATE",
    "ensure_directories",
]
