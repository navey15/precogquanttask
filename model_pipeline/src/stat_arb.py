"""Statistical arbitrage helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import coint


@dataclass(slots=True)
class PairStudy:
    asset_a: str
    asset_b: str
    correlation: float
    pvalue: float
    hedge_ratio: float
    half_life: float
    spread: pd.Series
    zscore: pd.Series


def _wide_close(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        panel = panel.set_index(["Date", "Asset"]).sort_index()
    return panel["Close"].unstack("Asset").sort_index()


def identify_correlated_pairs(panel: pd.DataFrame, top_n: int = 25, lookback: int = 252) -> List[tuple[str, str, float]]:
    close = _wide_close(panel)
    if lookback:
        close = close.iloc[-lookback:]
    returns = close.pct_change().dropna(how="all")
    corr = returns.corr().stack()
    corr = corr[corr.index.get_level_values(0) < corr.index.get_level_values(1)]
    top = corr.sort_values(ascending=False).head(top_n)
    return [(a, b, float(val)) for (a, b), val in top.items()]


def _hedge_ratio(series_a: pd.Series, series_b: pd.Series) -> float:
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if len(aligned) < 50:
        return np.nan
    y = aligned.iloc[:, 0]
    x = add_constant(aligned.iloc[:, 1])
    model = OLS(y, x).fit()
    return float(model.params.iloc[1])


def _half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 50:
        return np.nan
    y = spread.diff().dropna()
    x = spread.shift(1).dropna().loc[y.index]
    x = add_constant(x)
    model = OLS(y, x).fit()
    phi = model.params.iloc[1]
    if phi >= 0:
        return np.nan
    return float(-np.log(2) / phi)


def analyze_pair(panel: pd.DataFrame, asset_a: str, asset_b: str) -> PairStudy:
    close = _wide_close(panel)
    pair = close[[asset_a, asset_b]].dropna()
    corr = pair[asset_a].pct_change().corr(pair[asset_b].pct_change())
    score, pvalue, _ = coint(pair[asset_a], pair[asset_b])
    hedge = _hedge_ratio(pair[asset_a], pair[asset_b])
    spread = pair[asset_a] - hedge * pair[asset_b]
    zscore = (spread - spread.rolling(63).mean()) / (spread.rolling(63).std() + 1e-9)
    half_life = _half_life(spread)
    return PairStudy(
        asset_a=asset_a,
        asset_b=asset_b,
        correlation=float(corr),
        pvalue=float(pvalue),
        hedge_ratio=float(hedge),
        half_life=half_life,
        spread=spread,
        zscore=zscore,
    )


def generate_pair_signal(study: PairStudy, entry_z: float = 1.5, exit_z: float = 0.25) -> pd.DataFrame:
    df = pd.DataFrame({"spread": study.spread, "zscore": study.zscore})
    df["zscore"] = df["zscore"].ffill().fillna(0)
    z = df["zscore"]
    pos_values = []
    state = 0
    for value in z:
        if state == 0:
            if value > entry_z:
                state = -1
            elif value < -entry_z:
                state = 1
        else:
            if abs(value) < exit_z:
                state = 0
        pos_values.append(state)
    df["position"] = pd.Series(pos_values, index=z.index, dtype=float)
    return df


def scan_for_pairs(panel: pd.DataFrame, top_n: int = 5) -> List[PairStudy]:
    candidates = identify_correlated_pairs(panel, top_n=top_n * 4)
    studies: List[PairStudy] = []
    for asset_a, asset_b, _ in candidates:
        study = analyze_pair(panel, asset_a, asset_b)
        if np.isfinite(study.pvalue) and study.pvalue < 0.05:
            studies.append(study)
        if len(studies) >= top_n:
            break
    return studies


__all__ = [
    "PairStudy",
    "identify_correlated_pairs",
    "analyze_pair",
    "generate_pair_signal",
    "scan_for_pairs",
]
