"""
SEC EDGAR HTTP client.

Handles CIK lookups, filing index retrieval, and filing-text download/parsing.
Respects SEC rate limits (10 requests/sec) and User-Agent requirements.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Any

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state for rate-limiting
# ---------------------------------------------------------------------------
_last_request_time: float = 0.0


def _rate_limited_get(url: str, headers: dict | None = None) -> requests.Response:
    """GET with SEC-compliant rate limiting and User-Agent."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < config.SEC_RATE_LIMIT:
        time.sleep(config.SEC_RATE_LIMIT - elapsed)

    hdrs = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    if headers:
        hdrs.update(headers)

    resp = requests.get(url, headers=hdrs, timeout=30)
    _last_request_time = time.time()
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------
_cik_cache: dict[str, str] = {}


def get_cik(ticker: str) -> str | None:
    """Return zero-padded 10-digit CIK string for *ticker*, or None."""
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    try:
        data: dict[str, Any] = _rate_limited_get(config.SEC_TICKERS_URL).json()
    except Exception:
        logger.exception("Failed to fetch SEC tickers JSON")
        return None

    for entry in data.values():
        t = str(entry.get("ticker", "")).upper()
        cik = str(entry.get("cik_str", ""))
        _cik_cache[t] = cik.zfill(10)

    return _cik_cache.get(ticker)


# ---------------------------------------------------------------------------
# Filing index retrieval
# ---------------------------------------------------------------------------

def get_recent_filings(
    cik: str,
    form_types: list[str] | None = None,
    max_results: int = 40,
) -> list[dict[str, str]]:
    """
    Return a list of recent filings for *cik*.

    Each dict: {"accession", "form", "date", "primaryDocument"}.
    Filtered to *form_types* if provided.
    """
    if form_types is None:
        form_types = config.SEC_FORM_TYPES

    url = f"{config.SEC_BASE_URL}/submissions/CIK{cik}.json"
    try:
        data = _rate_limited_get(url).json()
    except Exception:
        logger.exception("Failed to fetch submissions for CIK %s", cik)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    results: list[dict[str, str]] = []
    for form, date, acc, doc in zip(forms, dates, accessions, docs):
        if form in form_types:
            results.append({
                "accession": acc,
                "form": form,
                "date": date,
                "primaryDocument": doc,
            })
        if len(results) >= max_results:
            break

    return results


# ---------------------------------------------------------------------------
# Filing text download & parse
# ---------------------------------------------------------------------------

def _build_filing_url(cik: str, accession: str, primary_doc: str) -> str:
    """Construct the full URL to a specific filing document."""
    acc_no_dash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no_dash}/{primary_doc}"


def download_filing_text(cik: str, filing: dict[str, str]) -> str | None:
    """
    Download a filing and return cleaned plain text, or None on failure.

    Tries the primary document first.  Falls back to the filing index page
    and looks for an .htm/.html link if the primary document is not HTML.
    """
    url = _build_filing_url(cik, filing["accession"], filing["primaryDocument"])
    try:
        resp = _rate_limited_get(url)
    except Exception:
        logger.warning("Failed to download filing: %s", url)
        return None

    text = _html_to_text(resp.text)
    if text and len(text.split()) >= config.MIN_FILING_WORDS:
        return text

    # Fallback: try the index page for an alternative HTML document
    acc_no_dash = filing["accession"].replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik.lstrip('0')}/{acc_no_dash}/"
    )
    try:
        idx_resp = _rate_limited_get(index_url)
        soup = BeautifulSoup(idx_resp.text, "lxml")
        for link in soup.find_all("a", href=True):
            href: str = link["href"]
            if href.endswith((".htm", ".html")) and href != filing["primaryDocument"]:
                alt_url = f"https://www.sec.gov{href}" if href.startswith("/") else f"{index_url}{href}"
                alt_resp = _rate_limited_get(alt_url)
                alt_text = _html_to_text(alt_resp.text)
                if alt_text and len(alt_text.split()) >= config.MIN_FILING_WORDS:
                    return alt_text
    except Exception:
        logger.debug("Fallback filing fetch failed for %s", index_url)

    return text if text else None


def _html_to_text(html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script / style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Convenience: fetch filings for a ticker around a specific date
# ---------------------------------------------------------------------------

def get_filings_near_date(
    ticker: str,
    target_date: str,
    window_days: int = 30,
    form_types: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Return filings for *ticker* filed within *window_days* of *target_date*.

    *target_date* in "YYYY-MM-DD" format.
    """
    from datetime import datetime, timedelta

    cik = get_cik(ticker)
    if cik is None:
        logger.warning("CIK not found for %s", ticker)
        return []

    filings = get_recent_filings(cik, form_types=form_types)
    target = datetime.strptime(target_date, "%Y-%m-%d")
    window = timedelta(days=window_days)

    matched: list[dict[str, str]] = []
    for f in filings:
        try:
            fdate = datetime.strptime(f["date"], "%Y-%m-%d")
        except ValueError:
            continue
        if abs(fdate - target) <= window:
            matched.append(f)

    return matched
