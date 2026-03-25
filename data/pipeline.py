"""
Data-collection pipeline.

Orchestrates calls to ``edgar_client`` and ``price_client``, persists raw data
to ``outputs/raw_data/{TICKER}/{DATE}/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from data import edgar_client, price_client

logger = logging.getLogger(__name__)


def _event_dir(ticker: str, event_date: str) -> Path:
    """Return (and create) the output directory for one event."""
    d = config.RAW_DATA_DIR / ticker.upper() / event_date
    d.mkdir(parents=True, exist_ok=True)
    return d


def collect_all(
    tickers: list[str] | None = None,
    max_quarters: int | None = None,
    force: bool = False,
) -> list[dict[str, str]]:
    """
    Collect SEC filings and price data for every ticker × earnings event.

    Parameters
    ----------
    tickers : list[str] | None
        Override ``config.TICKERS``.
    max_quarters : int | None
        Override ``config.MAX_QUARTERS``.
    force : bool
        If True, re-download even when local files already exist.

    Returns
    -------
    list of dicts ``{"ticker", "event_date", "has_text", "has_prices"}``
    summarising what was collected.
    """
    if tickers is None:
        tickers = config.TICKERS
    if max_quarters is None:
        max_quarters = config.MAX_QUARTERS

    config.ensure_output_dirs()
    manifest: list[dict[str, str]] = []

    for ticker in tqdm(tickers, desc="Tickers"):
        logger.info("Processing %s …", ticker)

        # --- earnings dates ---
        dates = price_client.get_earnings_dates(ticker, max_quarters=max_quarters)
        if not dates:
            logger.warning("No earnings dates for %s — skipping", ticker)
            continue

        for edate in tqdm(dates, desc=f"  {ticker} events", leave=False):
            out = _event_dir(ticker, edate)
            text_path = out / "filing.txt"
            price_path = out / "prices.csv"

            has_text = False
            has_prices = False

            # --- filing text ---
            if force or not text_path.exists():
                text = _fetch_filing_text(ticker, edate)
                if text:
                    text_path.write_text(text, encoding="utf-8")
                    has_text = True
                    logger.info("  %s %s — filing saved (%d words)",
                                ticker, edate, len(text.split()))
                else:
                    logger.warning("  %s %s — no filing text found", ticker, edate)
            else:
                has_text = text_path.exists() and text_path.stat().st_size > 0

            # --- price data ---
            if force or not price_path.exists():
                prices = price_client.get_price_data(ticker, edate)
                if prices is not None and not prices.empty:
                    prices.to_csv(price_path)
                    has_prices = True
                    logger.info("  %s %s — prices saved (%d rows)",
                                ticker, edate, len(prices))
                else:
                    logger.warning("  %s %s — no price data", ticker, edate)
            else:
                has_prices = price_path.exists() and price_path.stat().st_size > 0

            manifest.append({
                "ticker": ticker,
                "event_date": edate,
                "has_text": str(has_text),
                "has_prices": str(has_prices),
            })

    # Save manifest
    mf = pd.DataFrame(manifest)
    mf.to_csv(config.RAW_DATA_DIR / "manifest.csv", index=False)
    logger.info("Collection complete — %d events across %d tickers",
                len(manifest), len(tickers))
    return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_filing_text(ticker: str, event_date: str) -> str | None:
    """Try to get filing text near the earnings date."""
    cik = edgar_client.get_cik(ticker)
    if cik is None:
        return None

    filings = edgar_client.get_filings_near_date(
        ticker, event_date, window_days=30,
    )
    if not filings:
        # Widen window as fallback
        filings = edgar_client.get_filings_near_date(
            ticker, event_date, window_days=60,
        )

    if not filings:
        return None

    # Prefer 8-K (closest to earnings), then 10-Q
    for form_type in ("8-K", "10-Q"):
        for f in filings:
            if f["form"] == form_type:
                text = edgar_client.download_filing_text(cik, f)
                if text and len(text.split()) >= config.MIN_FILING_WORDS:
                    return text
    return None
