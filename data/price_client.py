"""
Yahoo Finance client for earnings dates and OHLCV price data.

Uses the ``yfinance`` library.  All date handling uses timezone-naive dates
to avoid subtle comparison bugs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Earnings dates
# ---------------------------------------------------------------------------

def get_earnings_dates(ticker: str, max_quarters: int | None = None) -> list[str]:
    """
    Return up to *max_quarters* most-recent past earnings dates for *ticker*.

    Dates are returned as ``"YYYY-MM-DD"`` strings, sorted oldest-first.
    Future dates are excluded.
    """
    if max_quarters is None:
        max_quarters = config.MAX_QUARTERS

    t = yf.Ticker(ticker)
    try:
        # yfinance returns a DataFrame indexed by earnings dates
        df = t.get_earnings_dates(limit=20)
    except Exception:
        logger.exception("Failed to get earnings dates for %s", ticker)
        return []

    if df is None or df.empty:
        logger.warning("No earnings dates returned for %s", ticker)
        return []

    today = pd.Timestamp(datetime.now().date())
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    past_dates = sorted(d for d in idx if d <= today)

    # Take most recent N
    selected = past_dates[-max_quarters:]
    return [d.strftime("%Y-%m-%d") for d in selected]


# ---------------------------------------------------------------------------
# OHLCV price data
# ---------------------------------------------------------------------------

def get_price_data(
    ticker: str,
    event_date: str,
    pre_days: int | None = None,
    post_days: int | None = None,
) -> pd.DataFrame | None:
    """
    Download daily OHLCV data for *ticker* in a window around *event_date*.

    Parameters
    ----------
    ticker : str
    event_date : str  – "YYYY-MM-DD"
    pre_days / post_days : int
        Calendar-day offsets (converted from trading-day config defaults).
        We request extra calendar days to account for weekends/holidays.

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume] indexed by
    timezone-naive DatetimeIndex, or None on failure.
    """
    if pre_days is None:
        # Convert trading days to approximate calendar days (×1.5 buffer)
        pre_days = int(abs(config.EVENT_WINDOW[0]) * 1.5)
    if post_days is None:
        post_days = int(abs(config.EVENT_WINDOW[1]) * 1.5)

    centre = datetime.strptime(event_date, "%Y-%m-%d")
    start = (centre - timedelta(days=pre_days)).strftime("%Y-%m-%d")
    end = (centre + timedelta(days=post_days)).strftime("%Y-%m-%d")

    t = yf.Ticker(ticker)
    try:
        df: pd.DataFrame = t.history(start=start, end=end, auto_adjust=True)
    except Exception:
        logger.exception("Failed to download price data for %s", ticker)
        return None

    if df is None or df.empty:
        logger.warning("Empty price data for %s around %s", ticker, event_date)
        return None

    # Ensure timezone-naive index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep standard OHLCV columns only
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols]
