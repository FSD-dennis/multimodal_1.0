"""
Feature fusion: merge text and price features into a single matrix.

Handles missing values (median imputation + boolean flags) and writes
``outputs/features/feature_matrix.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from features import text_features, price_features

logger = logging.getLogger(__name__)


def build_feature_matrix(raw_data_dir: Path | None = None) -> pd.DataFrame:
    """
    Walk ``raw_data_dir`` (default ``outputs/raw_data/``), extract features for
    every event, and return a merged DataFrame.

    The DataFrame is also saved to ``outputs/features/feature_matrix.csv``.
    """
    if raw_data_dir is None:
        raw_data_dir = config.RAW_DATA_DIR

    config.ensure_output_dirs()
    rows: list[dict] = []

    manifest_path = raw_data_dir / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        # Build manifest by scanning directory structure
        manifest = _scan_manifest(raw_data_dir)

    if manifest.empty:
        logger.error("No events found in %s", raw_data_dir)
        return pd.DataFrame()

    for _, row in tqdm(
        manifest.iterrows(), total=len(manifest), desc="Extracting features"
    ):
        ticker = row["ticker"]
        event_date = row["event_date"]
        event_dir = raw_data_dir / ticker / event_date

        record: dict = {"ticker": ticker, "event_date": event_date}

        # --- Text features ---
        text_path = event_dir / "filing.txt"
        text_missing = True
        if text_path.exists() and text_path.stat().st_size > 0:
            text = text_path.read_text(encoding="utf-8")
            if len(text.split()) >= config.MIN_FILING_WORDS:
                try:
                    tf = text_features.extract_all_text_features(text)
                    record.update(tf)
                    text_missing = False
                except Exception:
                    logger.exception("Text feature extraction failed for %s %s",
                                     ticker, event_date)

        record["text_missing"] = text_missing

        # Fill text feature columns with NaN if missing
        if text_missing:
            for col in config.TEXT_FEATURES:
                record.setdefault(col, np.nan)

        # --- Price features ---
        price_path = event_dir / "prices.csv"
        if price_path.exists() and price_path.stat().st_size > 0:
            try:
                prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
                pf = price_features.extract_all_price_features(prices, event_date)
                record.update(pf)
            except Exception:
                logger.exception("Price feature extraction failed for %s %s",
                                 ticker, event_date)
                for col in config.PRICE_FEATURES:
                    record.setdefault(col, np.nan)
        else:
            for col in config.PRICE_FEATURES:
                record.setdefault(col, np.nan)

        rows.append(record)

    df = pd.DataFrame(rows)

    if df.empty:
        logger.error("Feature matrix is empty — no data was extracted")
        return df

    # --- Missing-value imputation ---
    df = _impute_missing(df)

    # --- Sort chronologically ---
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    # --- Save ---
    out_path = config.FEATURES_DIR / "feature_matrix.csv"
    df.to_csv(out_path, index=False)
    logger.info("Feature matrix saved → %s  (%d rows × %d cols)",
                out_path, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_manifest(raw_data_dir: Path) -> pd.DataFrame:
    """Build a manifest by scanning the directory tree."""
    records = []
    for ticker_dir in sorted(raw_data_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for date_dir in sorted(ticker_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            records.append({
                "ticker": ticker_dir.name,
                "event_date": date_dir.name,
                "has_text": str((date_dir / "filing.txt").exists()),
                "has_prices": str((date_dir / "prices.csv").exists()),
            })
    return pd.DataFrame(records)


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric feature values with column medians.

    The ``text_missing`` boolean flag is preserved as-is.
    """
    numeric_cols = [
        c for c in df.columns
        if c not in config.META_COLUMNS + ["text_missing"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed %d missing values in '%s' with median %.6f",
                        n_missing, col, median_val)
    return df
