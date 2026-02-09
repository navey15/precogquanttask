"""Alpha modeling helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR, ensure_directories

TARGET_COLUMN = "fwd_return_5d"


@dataclass(slots=True)
class SplitBounds:
    train_end: pd.Timestamp
    val_end: pd.Timestamp


@dataclass(slots=True)
class ModelResult:
    pipeline: Pipeline
    feature_columns: List[str]
    target_column: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    feature_importances: pd.Series
    predictions: pd.Series


def infer_feature_columns(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> List[str]:
    drop_cols = {
        target,
        "fwd_return_1d",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }
    return [col for col in frame.columns if col not in drop_cols]


def compute_temporal_split(frame: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> SplitBounds:
    dates = frame.index.get_level_values("Date").unique()
    n = len(dates)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    return SplitBounds(train_end=dates[train_idx - 1], val_end=dates[val_idx - 1])


def split_frame(frame: pd.DataFrame, bounds: SplitBounds) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx_dates = frame.index.get_level_values("Date")
    train = frame[idx_dates <= bounds.train_end]
    val = frame[(idx_dates > bounds.train_end) & (idx_dates <= bounds.val_end)]
    test = frame[idx_dates > bounds.val_end]
    return train, val, test


def _ic_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    corr, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(corr)


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=400,
                    l2_regularization=0.1,
                    min_samples_leaf=50,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(frame: pd.DataFrame, random_state: int = 42) -> ModelResult:
    feature_columns = infer_feature_columns(frame)
    bounds = compute_temporal_split(frame)
    train, val, test = split_frame(frame, bounds)

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(train[feature_columns], train[TARGET_COLUMN])

    def _metrics(split: pd.DataFrame) -> Dict[str, float]:
        preds = pipeline.predict(split[feature_columns])
        mse = mean_squared_error(split[TARGET_COLUMN], preds)
        ic = _ic_score(split[TARGET_COLUMN].to_numpy(), preds)
        return {"rmse": float(np.sqrt(mse)), "ic": ic}

    train_metrics = _metrics(train)
    val_metrics = _metrics(val)

    all_preds = pipeline.predict(frame[feature_columns])
    predictions = pd.Series(all_preds, index=frame.index, name="alpha_signal")

    model = pipeline.named_steps["model"]
    try:
        raw_importance = model.feature_importances_
    except AttributeError:
        # HistGradientBoostingRegressor lacks feature_importances_ in older
        # sklearn releases; use permutation importance for interpretability.
        from sklearn.inspection import permutation_importance

        perm = permutation_importance(
            pipeline,
            val[feature_columns],
            val[TARGET_COLUMN],
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
        )
        raw_importance = perm.importances_mean

    feature_importances = pd.Series(raw_importance, index=feature_columns)

    ensure_directories()
    model_path = MODEL_DIR / "hist_gbr_model.pkl"
    import joblib  # lazy import to avoid dependency if unused

    joblib.dump(pipeline, model_path)

    return ModelResult(
        pipeline=pipeline,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        feature_importances=feature_importances,
        predictions=predictions,
    )


def attach_predictions(frame: pd.DataFrame, preds: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    frame["alpha_signal"] = preds
    return frame


def smooth_alpha_signal(
    alphas: pd.DataFrame,
    smoothing_window: int = 5,
    signal_col: str = "alpha_signal",
) -> pd.DataFrame:
    """Applies a rolling average to the alpha signal to reduce noise."""
    alphas = alphas.copy()
    smoothed_signal = alphas.groupby("Asset")[signal_col].transform(
        lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
    )
    alphas[f"{signal_col}_smoothed"] = smoothed_signal
    return alphas


__all__ = [
    "SplitBounds",
    "ModelResult",
    "infer_feature_columns",
    "compute_temporal_split",
    "split_frame",
    "build_pipeline",
    "train_model",
    "attach_predictions",
    "smooth_alpha_signal",
]
