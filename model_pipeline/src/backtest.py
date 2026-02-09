"""Portfolio construction and backtesting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import ANNUAL_TRADING_DAYS, DEFAULT_COST_BPS, RISK_FREE_RATE


@dataclass(slots=True)
class BacktestConfig:
    top_quantile: float = 0.2
    gross_leverage: float = 1.5
    cost_bps: float = DEFAULT_COST_BPS
    rebalance_frequency: str = 'D'  # 'D' for daily, 'W' for weekly
    signal_column: str = "alpha_signal"
    return_column: str = "fwd_return_1d"
    risk_column: str = "ann_vol_21d"


def _normalize_weights(weights: pd.Series, gross_leverage: float) -> pd.Series:
    if weights.abs().sum() == 0:
        return weights
    weights = weights / weights.abs().sum()
    return weights * gross_leverage


def score_to_weights(frame: pd.DataFrame, cfg: BacktestConfig | None = None) -> pd.DataFrame:
    cfg = cfg or BacktestConfig()
    frame = frame.copy()

    def _alloc(group: pd.DataFrame) -> pd.Series:
        scores = group[cfg.signal_column]
        if scores.nunique() < 2:
            return pd.Series(0.0, index=group.index)
        long_cut = scores.quantile(1 - cfg.top_quantile)
        short_cut = scores.quantile(cfg.top_quantile)
        weights = pd.Series(0.0, index=group.index)
        long_mask = scores >= long_cut
        short_mask = scores <= short_cut
        inv_vol = 1.0 / (group[cfg.risk_column].replace(0, np.nan) + 1e-6)
        weights.loc[long_mask] = inv_vol.loc[long_mask]
        weights.loc[short_mask] = -inv_vol.loc[short_mask]
        weights -= weights.mean()  # enforce (approx) dollar neutrality before scaling
        weights = _normalize_weights(weights, cfg.gross_leverage)
        return weights

    if cfg.rebalance_frequency == 'W':
        # Identify the start of each week
        rebalance_dates = frame.index.get_level_values('Date').to_series().groupby(pd.Grouper(freq='W-MON')).first().unique()
        
        # Calculate weights only on rebalance dates
        rebalance_frame = frame.loc[rebalance_dates]
        rebalance_weights = rebalance_frame.groupby('Date', group_keys=False).apply(_alloc)
        
        # Forward-fill weights to all days
        all_days_index = frame.index.droplevel('Asset').unique()
        w_unstacked = rebalance_weights.unstack('Asset')
        w_unstacked = w_unstacked.reindex(all_days_index, method='ffill')
        weights = w_unstacked.stack('Asset')
    else:
        # Daily rebalance
        weights = frame.groupby("Date", group_keys=False).apply(_alloc)

    weights.name = "weight"
    return weights


def _cost_rate(cost_bps: float) -> float:
    return cost_bps / 10000


def run_backtest(frame: pd.DataFrame, cfg: BacktestConfig | None = None) -> Dict[str, pd.DataFrame | pd.Series | Dict[str, float]]:
    cfg = cfg or BacktestConfig()
    weights = score_to_weights(frame, cfg)
    # Convert to wide matrices for vectorised math
    w_wide = weights.unstack("Asset").fillna(0.0)
    r_wide = frame[cfg.return_column].unstack("Asset").reindex_like(w_wide)
    r_wide = r_wide.fillna(0.0)

    portfolio_returns = (w_wide * r_wide).sum(axis=1)
    turnover = 0.5 * w_wide.diff().abs().sum(axis=1)
    turnover = turnover.fillna(0.0)
    cost = turnover * _cost_rate(cfg.cost_bps)
    net_returns = portfolio_returns - cost

    eq_curve = (1 + net_returns).cumprod()
    benchmark = r_wide.mean(axis=1)
    benchmark_curve = (1 + benchmark).cumprod()

    metrics = compute_performance_metrics(net_returns, benchmark, turnover_series=turnover)

    results = pd.DataFrame(
        {
            "portfolio_ret": net_returns,
            "gross_ret": portfolio_returns,
            "benchmark_ret": benchmark,
            "turnover": turnover,
            "cost": cost,
            "equity_curve": eq_curve,
            "benchmark_curve": benchmark_curve,
        }
    )

    return {"weights": weights, "timeline": results, "metrics": metrics}


def compute_performance_metrics(
    returns: pd.Series,
    benchmark: pd.Series,
    turnover_series: pd.Series | None = None,
) -> Dict[str, float]:
    ann_factor = np.sqrt(ANNUAL_TRADING_DAYS)
    sharpe = ((returns.mean() - RISK_FREE_RATE / ANNUAL_TRADING_DAYS) / returns.std()) * ann_factor if returns.std() > 0 else np.nan
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1
    max_dd = drawdown.min()
    avg_dd = drawdown.mean()
    avg_turnover = turnover_series.mean() if turnover_series is not None else np.nan
    total_return = cum.iloc[-1] - 1 if len(cum) else np.nan
    rel_return = cum.iloc[-1] - (1 + benchmark).cumprod().iloc[-1] if len(cum) else np.nan
    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_drawdown": float(avg_dd),
        "total_return": float(total_return),
        "relative_return": float(rel_return),
        "avg_turnover": float(avg_turnover),
    }


def run_long_short_backtest(
    feature_panel: pd.DataFrame,
    alpha_panel: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> Dict[str, pd.DataFrame | pd.Series | Dict[str, float]]:
    """Convenience wrapper that merges features with alpha signals before running the engine."""

    cfg = config or BacktestConfig()

    def _as_frame(panel: pd.DataFrame) -> pd.DataFrame:
        if isinstance(panel.index, pd.MultiIndex):
            return panel.reset_index()
        return panel.copy()

    feature_panel = _as_frame(feature_panel)
    alpha_panel = _as_frame(alpha_panel)
    if "date" in alpha_panel.columns and "Date" not in alpha_panel.columns:
        alpha_panel = alpha_panel.rename(columns={"date": "Date"})

    merge_cols = ["Date", "Asset"]
    if cfg.signal_column not in alpha_panel.columns:
        raise ValueError(f"Alpha panel must contain '{cfg.signal_column}' column.")

    merged = feature_panel.merge(alpha_panel[merge_cols + [cfg.signal_column]], on=merge_cols, how="inner")
    merged = merged.set_index(merge_cols).sort_index()
    return run_backtest(merged, cfg)


__all__ = [
    "BacktestConfig",
    "score_to_weights",
    "run_backtest",
    "run_long_short_backtest",
    "compute_performance_metrics",
]
