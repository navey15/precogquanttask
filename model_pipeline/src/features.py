"""Feature engineering utilities."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import ANNUAL_TRADING_DAYS, FEATURE_STORE, ensure_directories


@dataclass(slots=True)
class FeatureConfig:
    momentum_windows: Sequence[int] = (1, 3, 5, 10, 21, 63)
    volatility_windows: Sequence[int] = (5, 10, 21, 63)
    volume_windows: Sequence[int] = (5, 20)
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


def _group(panel: pd.DataFrame):
    return panel.groupby("Asset", group_keys=False)


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi



def _cross_sectional_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("Date")[column]
    demeaned = frame[column] - grouped.transform("mean")
    denom = grouped.transform("std").replace(0, np.nan)
    return demeaned / (denom + 1e-9)


def build_feature_matrix(panel: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Create alpha factors + targets from the cleaned price panel."""
    cfg = cfg or FeatureConfig()
    panel = panel.copy()
    panel = panel.sort_values(["Asset", "Date"]).reset_index(drop=True)
    g = _group(panel)

    for window in cfg.momentum_windows:
        panel[f"ret_{window}d"] = g["Close"].pct_change(window)
        panel[f"mom_{window}d"] = g["Close"].transform(lambda s, w=window: s / s.shift(w) - 1)

    daily_ret = g["Close"].pct_change()
    for window in cfg.volatility_windows:
        panel[f"ann_vol_{window}d"] = (
            daily_ret.groupby(panel["Asset"]).transform(lambda s, w=window: s.rolling(w).std())
            * np.sqrt(ANNUAL_TRADING_DAYS)
        )

    for window in cfg.volume_windows:
        rolling_volume = g["Volume"].transform(lambda s, w=window: s.rolling(w).mean())
        panel[f"vol_z_{window}d"] = (panel["Volume"] - rolling_volume) / (rolling_volume + 1e-9)

    panel["rsi"] = g["Close"].transform(_compute_rsi, period=cfg.rsi_period)

    prev_close = g["Close"].shift(1)
    true_range = pd.concat([
        panel["High"] - panel["Low"],
        (panel["High"] - prev_close).abs(),
        (panel["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    panel["atr"] = true_range.groupby(panel["Asset"]).transform(
        lambda s: s.rolling(cfg.atr_period).mean()
    )

    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    ema_fast = g["Close"].transform(_ema, span=cfg.macd_fast)
    ema_slow = g["Close"].transform(_ema, span=cfg.macd_slow)
    panel["macd"] = ema_fast - ema_slow
    panel["macd_signal"] = panel.groupby("Asset")["macd"].transform(_ema, span=cfg.macd_signal)
    panel["macd_hist"] = panel["macd"] - panel["macd_signal"]

    rolling_mean_20 = g["Close"].transform(lambda s: s.rolling(20).mean())
    rolling_std_20 = g["Close"].transform(lambda s: s.rolling(20).std())
    panel["price_distance_20"] = (panel["Close"] - rolling_mean_20) / (rolling_std_20 + 1e-9)

    range_width = (panel["High"] - panel["Low"]).replace(0, np.nan)
    panel["close_location"] = (panel["Close"] - panel["Low"]) / (range_width + 1e-9) - 0.5

    panel["intraday_range"] = (panel["High"] / panel["Low"]) - 1

    # Momentum and Mean-Reversion Factors
    for window in [21, 63, 126, 252]:  # 1-month, 3-month, 6-month, 1-year
        panel[f"momentum_{window}d"] = g["Close"].pct_change(window)

    # Cross-sectional context
    panel["cs_ret_1d_z"] = _cross_sectional_zscore(panel, "ret_1d")
    panel["cs_vol_21d_z"] = _cross_sectional_zscore(panel, "ann_vol_21d")

    # Targets
    panel["fwd_return_1d"] = g["Close"].transform(lambda s: s.shift(-1) / s - 1)
    panel["fwd_return_5d"] = g["Close"].transform(lambda s: s.shift(-5) / s - 1)

    panel = panel.dropna().set_index(["Date", "Asset"]).sort_index()
    return panel


def persist_feature_matrix(features: pd.DataFrame, engine: str | None = None) -> None:
    """Persist the factor store, trying both parquet engines for robustness."""
    ensure_directories()
    engines = [engine] if engine else ["pyarrow", "fastparquet"]
    errors: dict[str, Exception] = {}
    for eng in engines:
        if eng is None:
            continue
        try:
            features.to_parquet(FEATURE_STORE, engine=eng)
            return
        except Exception as exc:  # pragma: no cover - informative fallback only
            warnings.warn(f"Parquet write via '{eng}' failed ({exc}); trying fallback engine.")
            errors[eng] = exc
    detail = ", ".join(f"{k}: {v}" for k, v in errors.items()) or "unknown error"
    raise RuntimeError(
        "Unable to write feature store via any supported parquet engine. "
        f"Tried {list(errors.keys())}. Details: {detail}"
    )


__all__ = [
    "FeatureConfig",
    "build_feature_matrix",
    "persist_feature_matrix",
]
