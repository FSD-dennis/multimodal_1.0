"""
Price-feature engineering from OHLCV data around earnings events.

All functions take a ``pd.DataFrame`` of daily OHLCV data (with a
DatetimeIndex) and a target ``event_date`` string (YYYY-MM-DD).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(event_date: str) -> pd.Timestamp:
    return pd.Timestamp(event_date)


def _split_window(
    prices: pd.DataFrame,
    event_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prices into pre-event and post-event (inclusive of event day)."""
    ed = _parse_date(event_date)
    pre = prices.loc[prices.index < ed]
    post = prices.loc[prices.index >= ed]
    return pre, post


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def compute_pre_earnings_momentum(prices: pd.DataFrame, event_date: str) -> float | None:
    """Cumulative return over the pre-event window [-20, -1] trading days."""
    pre, _ = _split_window(prices, event_date)
    if pre.empty or "Close" not in pre.columns:
        return None
    returns = pre["Close"].pct_change().dropna()
    if returns.empty:
        return None
    return float((1 + returns).prod() - 1)


def compute_pre_earnings_volatility(prices: pd.DataFrame, event_date: str) -> float | None:
    """Realized volatility (std of daily log-returns) over the pre-event window."""
    pre, _ = _split_window(prices, event_date)
    if pre.empty or len(pre) < 3 or "Close" not in pre.columns:
        return None
    log_ret = np.log(pre["Close"] / pre["Close"].shift(1)).dropna()
    if log_ret.empty:
        return None
    return float(log_ret.std())


def compute_post_earnings_gap(prices: pd.DataFrame, event_date: str) -> float | None:
    """
    Overnight gap return on the earnings day.

    Computed as  Open[event_day] / Close[previous_day] - 1.
    Falls back to intra-day return if previous close is unavailable.
    """
    pre, post = _split_window(prices, event_date)
    if post.empty or "Open" not in post.columns:
        return None

    event_open = post.iloc[0]["Open"]
    if not pre.empty and "Close" in pre.columns:
        prev_close = pre.iloc[-1]["Close"]
        if prev_close != 0:
            return float(event_open / prev_close - 1)

    # Fallback: intra-day return on event day
    if "Close" in post.columns:
        event_close = post.iloc[0]["Close"]
        if event_open != 0:
            return float(event_close / event_open - 1)
    return None


def compute_volume_spike(prices: pd.DataFrame, event_date: str) -> float | None:
    """Ratio of event-day volume to average pre-event volume."""
    pre, post = _split_window(prices, event_date)
    if post.empty or "Volume" not in post.columns:
        return None
    if pre.empty or "Volume" not in pre.columns:
        return None

    avg_vol = pre["Volume"].mean()
    if avg_vol == 0:
        return None
    return float(post.iloc[0]["Volume"] / avg_vol)


def compute_amihud_illiquidity(prices: pd.DataFrame, event_date: str) -> float | None:
    """
    Amihud (2002) illiquidity proxy over the pre-event window.

    Illiquidity = mean( |daily_return| / dollar_volume ).
    Higher values indicate less liquid stocks.
    """
    pre, _ = _split_window(prices, event_date)
    if pre.empty or len(pre) < 3:
        return None
    if "Close" not in pre.columns or "Volume" not in pre.columns:
        return None

    ret = pre["Close"].pct_change().abs().dropna()
    vol = pre["Volume"].iloc[1:]  # align with returns
    dollar_vol = pre["Close"].iloc[1:] * vol

    mask = dollar_vol > 0
    if mask.sum() == 0:
        return None
    ratio = ret[mask] / dollar_vol[mask]
    return float(ratio.mean())


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

def compute_targets(
    prices: pd.DataFrame,
    event_date: str,
    horizons: list[int] | None = None,
) -> dict[str, float | None]:
    """
    Compute post-earnings returns for each horizon.

    Returns dict like ``{"ret_1d": 0.023, "ret_3d": -0.011}``.
    """
    if horizons is None:
        horizons = config.TARGET_HORIZONS

    _, post = _split_window(prices, event_date)
    if post.empty or "Close" not in post.columns:
        return {f"ret_{h}d": None for h in horizons}

    base_close = post.iloc[0]["Close"]
    if base_close == 0:
        return {f"ret_{h}d": None for h in horizons}

    targets: dict[str, float | None] = {}
    for h in horizons:
        if len(post) > h:
            targets[f"ret_{h}d"] = float(post.iloc[h]["Close"] / base_close - 1)
        else:
            targets[f"ret_{h}d"] = None
    return targets


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

def extract_all_price_features(
    prices: pd.DataFrame,
    event_date: str,
) -> dict[str, float | None]:
    """
    Extract all price features and targets for one event.

    Returns a flat dict containing all ``config.PRICE_FEATURES`` keys
    plus ``ret_1d``, ``ret_3d``, etc.
    """
    features: dict[str, float | None] = {
        "pre_earnings_momentum": compute_pre_earnings_momentum(prices, event_date),
        "pre_earnings_volatility": compute_pre_earnings_volatility(prices, event_date),
        "post_earnings_gap": compute_post_earnings_gap(prices, event_date),
        "volume_spike": compute_volume_spike(prices, event_date),
        "amihud_illiquidity": compute_amihud_illiquidity(prices, event_date),
    }
    features.update(compute_targets(prices, event_date))
    return features
