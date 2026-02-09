"""Data access and cleaning helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def list_price_files(data_dir: Path | str = DATA_DIR) -> List[Path]:
    """Return every CSV file that represents an asset's history."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} is missing")
    return sorted(p for p in data_dir.glob("Asset_*.csv") if p.is_file())


def _load_single_asset(path: Path) -> pd.DataFrame:
    asset_id = path.stem
    df = pd.read_csv(path, parse_dates=["Date"])  # type: ignore[arg-type]
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path.name} missing columns: {missing_cols}")
    df = df[REQUIRED_COLUMNS].copy()
    df["Asset"] = asset_id
    return df


def load_price_panel(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load every asset file into a single tidy DataFrame."""
    frames = [_load_single_asset(path) for path in list_price_files(data_dir)]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["Asset", "Date"]).reset_index(drop=True)
    return panel


def assess_data_quality(panel: pd.DataFrame) -> Dict[str, object]:
    """Provide quick stats that highlight data health."""
    panel = panel.copy()
    summary = {
        "rows": len(panel),
        "assets": panel["Asset"].nunique(),
        "date_start": panel["Date"].min(),
        "date_end": panel["Date"].max(),
        "missing_ratio": panel.isna().mean().to_dict(),
        "zero_volume_days": int((panel["Volume"] <= 0).sum()),
    }
    return summary


def _winsorize_returns(panel: pd.DataFrame, z_threshold: float = 8.0) -> pd.DataFrame:
    """Clip extreme single-day moves that likely stem from bad prints."""
    panel = panel.copy()
    returns = panel.groupby("Asset")["Close"].pct_change()
    rolling_median = returns.groupby(panel["Asset"]).transform("median")
    mad = returns.groupby(panel["Asset"]).transform(lambda x: np.median(np.abs(x - np.median(x))) + 1e-9)
    z = (returns - rolling_median) / mad
    outlier_mask = z.abs() > z_threshold
    cols = ["Open", "High", "Low", "Close"]
    panel.loc[outlier_mask, cols] = np.nan
    return panel


def clean_price_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Handle missing data and mild anomalies before feature creation."""
    panel = panel.copy()
    panel = panel.drop_duplicates(subset=["Asset", "Date"]).sort_values(["Asset", "Date"])
    panel = _winsorize_returns(panel)

    def _fill_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("Date").asfreq("B")  # enforce business-day spacing
        g = g.interpolate(method="time").ffill().bfill()
        g["Volume"] = g["Volume"].mask(g["Volume"] <= 0)
        g["Volume"] = g["Volume"].interpolate(method="time").ffill().bfill()
        g = g.reset_index()
        g["Asset"] = g["Asset"].iloc[0]
        return g

    panel = panel.groupby("Asset", group_keys=False).apply(_fill_group)
    panel = panel.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return panel


def load_and_prepare_prices(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Convenience wrapper that loads and cleans the price panel."""
    raw = load_price_panel(data_dir)
    clean = clean_price_panel(raw)
    return clean


__all__ = [
    "list_price_files",
    "load_price_panel",
    "assess_data_quality",
    "clean_price_panel",
    "load_and_prepare_prices",
]
