"""
Model training with strict chronological split.

Trains multiple sklearn pipelines per horizon, selected via ``config.py``:
- Direction (classification): logistic, random_forest, gradient_boosting, svm, knn
- Magnitude (regression):     ridge, lasso, random_forest, gradient_boosting, svr, knn
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature columns used for modelling
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = config.TEXT_FEATURES + config.PRICE_FEATURES + ["text_missing"]


# ---------------------------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train / test by date — **no shuffle**.

    All rows in test have ``event_date`` strictly after every row in train.
    """
    if train_ratio is None:
        train_ratio = config.TRAIN_RATIO

    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    logger.info(
        "Chrono split → train: %d rows (up to %s)  |  test: %d rows (from %s)",
        len(train),
        train["event_date"].max().strftime("%Y-%m-%d") if len(train) else "N/A",
        len(test),
        test["event_date"].min().strftime("%Y-%m-%d") if len(test) else "N/A",
    )

    # Verify no leakage
    if len(train) and len(test):
        assert train["event_date"].max() <= test["event_date"].min(), (
            "TIME-SERIES LEAKAGE: train max date > test min date"
        )

    return train, test


# ---------------------------------------------------------------------------
# Pipeline builders — direction (classification)
# ---------------------------------------------------------------------------

def _build_direction_pipeline(model_key: str, seed: int) -> Pipeline:
    """
    Build a StandardScaler + classifier pipeline.

    Supported *model_key* values:
        logistic, random_forest, gradient_boosting, svm, knn
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    estimators = {
        "logistic": LogisticRegression(
            random_state=seed, max_iter=1000, class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=seed,
            class_weight="balanced", n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=seed,
        ),
        "svm": SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1,
        ),
    }

    if model_key not in estimators:
        raise ValueError(
            f"Unknown direction model '{model_key}'. "
            f"Choose from: {list(estimators.keys())}"
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimators[model_key]),
    ])


# ---------------------------------------------------------------------------
# Pipeline builders — magnitude (regression)
# ---------------------------------------------------------------------------

def _build_magnitude_pipeline(model_key: str, seed: int) -> Pipeline:
    """
    Build a StandardScaler + regressor pipeline.

    Supported *model_key* values:
        ridge, lasso, random_forest, gradient_boosting, svr, knn
    """
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor

    estimators = {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "lasso": Lasso(alpha=0.001, random_state=seed, max_iter=2000),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=seed,
        ),
        "svr": SVR(kernel="rbf"),
        "knn": KNeighborsRegressor(
            n_neighbors=5, weights="distance", n_jobs=-1,
        ),
    }

    if model_key not in estimators:
        raise ValueError(
            f"Unknown magnitude model '{model_key}'. "
            f"Choose from: {list(estimators.keys())}"
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", estimators[model_key]),
    ])


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_all_models(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, dict]:
    """
    Train direction + magnitude models for each target horizon.

    Models to train are read from ``config.DIRECTION_MODELS`` and
    ``config.MAGNITUDE_MODELS``.  Each combination of (horizon, model_key)
    produces one saved pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (output of ``fusion.build_feature_matrix``).
    output_dir : Path
        Where to save model files (default ``config.MODELS_DIR``).

    Returns
    -------
    dict mapping model names to dicts with keys
    ``{"pipeline", "train_df", "test_df", "target_col", "model_type",
      "model_key"}``.
    """
    if output_dir is None:
        output_dir = config.MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = chronological_split(df)

    # Ensure feature columns are numeric and clean
    for col in FEATURE_COLS:
        if col not in train_df.columns:
            train_df[col] = 0.0
            test_df[col] = 0.0
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0.0)

    results: dict[str, dict] = {}

    for horizon in config.TARGET_HORIZONS:
        target_col = f"ret_{horizon}d"
        if target_col not in df.columns:
            logger.warning("Target column '%s' not found — skipping", target_col)
            continue

        # --- Direction models (sign of return) ---
        for model_key in config.DIRECTION_MODELS:
            dir_name = f"direction_{horizon}d_{model_key}"
            dir_train = train_df.dropna(subset=[target_col]).copy()
            dir_test = test_df.dropna(subset=[target_col]).copy()

            if len(dir_train) < 5:
                logger.warning("Too few training samples for %s — skipping", dir_name)
                continue

            y_dir_train = (dir_train[target_col] >= 0).astype(int)
            X_dir_train = dir_train[FEATURE_COLS].values

            dir_pipe = _build_direction_pipeline(model_key, config.SEED)
            dir_pipe.fit(X_dir_train, y_dir_train)

            path = output_dir / f"{dir_name}.joblib"
            joblib.dump(dir_pipe, path)
            logger.info("Saved %s → %s", dir_name, path)

            results[dir_name] = {
                "pipeline": dir_pipe,
                "train_df": dir_train,
                "test_df": dir_test,
                "target_col": target_col,
                "model_type": "direction",
                "model_key": model_key,
            }

        # --- Magnitude models (continuous return) ---
        for model_key in config.MAGNITUDE_MODELS:
            mag_name = f"magnitude_{horizon}d_{model_key}"
            mag_train = train_df.dropna(subset=[target_col]).copy()
            mag_test = test_df.dropna(subset=[target_col]).copy()

            if len(mag_train) < 5:
                logger.warning("Too few training samples for %s — skipping", mag_name)
                continue

            y_mag_train = mag_train[target_col].values
            X_mag_train = mag_train[FEATURE_COLS].values

            mag_pipe = _build_magnitude_pipeline(model_key, config.SEED)
            mag_pipe.fit(X_mag_train, y_mag_train)

            path = output_dir / f"{mag_name}.joblib"
            joblib.dump(mag_pipe, path)
            logger.info("Saved %s → %s", mag_name, path)

            results[mag_name] = {
                "pipeline": mag_pipe,
                "train_df": mag_train,
                "test_df": mag_test,
                "target_col": target_col,
                "model_type": "magnitude",
                "model_key": model_key,
            }

    return results
