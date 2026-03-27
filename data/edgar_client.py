"""
SEC EDGAR HTTP client.

Handles CIK lookups, filing index retrieval, and filing-text download/parsing.
Respects SEC rate limits (10 requests/sec) and User-Agent requirements.

The main filing-retrieval function ``get_all_filings`` reads both the
``filings.recent`` block **and** the older-history JSON files listed under
``filings.files``, so it can find filings going back 5+ years — not just
the most recent batch.
"""

from __future__ import annotations

import re
import time
import logging
from datetime import datetime, timedelta
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
# Form-type matching helpers
# ---------------------------------------------------------------------------

def _matches_form_type(form: str, form_types: list[str]) -> bool:
    """
    Return True if *form* matches any entry in *form_types*.

    Handles amendments: if "8-K" is in form_types, both "8-K" and "8-K/A"
    will match.  You can also list "8-K/A" explicitly to match only the
    amendment.
    """
    form_upper = form.upper()
    for ft in form_types:
        ft_upper = ft.upper()
        if form_upper == ft_upper:
            return True
        # "8-K" also matches "8-K/A", "10-Q" also matches "10-Q/A", etc.
        if form_upper == ft_upper + "/A":
            return True
    return False


# ---------------------------------------------------------------------------
# Filing index retrieval — full history (recent + older archive files)
# ---------------------------------------------------------------------------

# Cache: CIK → full list of filings (already filtered & deduped).
# Avoids redundant HTTP calls when the same CIK is queried multiple times
# within a single pipeline run.
_filings_cache: dict[str, list[dict[str, str]]] = {}


def _parse_filing_block(block: dict[str, Any]) -> list[dict[str, str]]:
    """
    Parse one filing-array block (the structure shared by ``filings.recent``
    and each older-history JSON file) and return a list of filing dicts.

    Each dict: ``{"accession", "form", "date", "primaryDocument"}``.
    """
    forms = block.get("form", [])
    dates = block.get("filingDate", [])
    accessions = block.get("accessionNumber", [])
    docs = block.get("primaryDocument", [])

    results: list[dict[str, str]] = []
    for form, date, acc, doc in zip(forms, dates, accessions, docs):
        results.append({
            "accession": acc,
            "form": form,
            "date": date,
            "primaryDocument": doc,
        })
    return results


def get_all_filings(
    cik: str,
    form_types: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Return *all* filings for *cik*, spanning the full SEC history.

    How it works
    ------------
    1. Fetch the main submissions JSON at
       ``https://data.sec.gov/submissions/CIK{cik}.json``.
    2. Parse ``filings.recent`` (the most recent ~1 000 filings).
    3. Iterate ``filings.files`` — each entry points to an older-history
       JSON file (e.g. ``CIK0000320193-submissions-001.json``) hosted at
       the same base URL.  Fetch and parse each one.
    4. Merge everything, deduplicate by accession number, filter to
       *form_types*, and sort by filing date **descending** (newest first).

    Parameters
    ----------
    cik : str
        Zero-padded 10-digit CIK.
    form_types : list[str] | None
        Filing types to keep (e.g. ``["8-K", "10-Q"]``).  Amendments such
        as ``8-K/A`` are matched automatically.  If ``None``, defaults to
        ``config.SEC_FORM_TYPES``.

    Returns
    -------
    list[dict]
        Each dict has keys ``accession``, ``form``, ``date``,
        ``primaryDocument``.  Sorted newest-first.
    """
    if form_types is None:
        form_types = config.SEC_FORM_TYPES

    # Return cached result if available
    cache_key = cik
    if cache_key in _filings_cache:
        # Re-filter in case form_types differ from the cached run
        return [
            f for f in _filings_cache[cache_key]
            if _matches_form_type(f["form"], form_types)
        ]

    # --- Step 1: fetch main submissions JSON ---
    url = f"{config.SEC_BASE_URL}/submissions/CIK{cik}.json"
    try:
        data = _rate_limited_get(url).json()
    except Exception:
        logger.exception("Failed to fetch submissions for CIK %s", cik)
        return []

    filings_section = data.get("filings", {})

    # --- Step 2: parse the "recent" block ---
    all_raw = _parse_filing_block(filings_section.get("recent", {}))
    logger.debug("CIK %s — %d filings in 'recent' block", cik, len(all_raw))

    # --- Step 3: fetch and parse each older-history file ---
    older_files: list[dict[str, Any]] = filings_section.get("files", [])
    for file_info in older_files:
        fname = file_info.get("name", "")
        if not fname:
            continue
        hist_url = f"{config.SEC_BASE_URL}/submissions/{fname}"
        try:
            hist_data = _rate_limited_get(hist_url).json()
            older_filings = _parse_filing_block(hist_data)
            all_raw.extend(older_filings)
            logger.debug("CIK %s — %d filings from %s",
                         cik, len(older_filings), fname)
        except Exception:
            logger.warning("Failed to fetch older filings from %s", hist_url)

    # --- Step 4: deduplicate by accession number ---
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for f in all_raw:
        if f["accession"] not in seen:
            seen.add(f["accession"])
            unique.append(f)

    # --- Step 5: sort by date descending (newest first) ---
    def _sort_key(f: dict[str, str]) -> str:
        return f.get("date", "0000-00-00")
    unique.sort(key=_sort_key, reverse=True)

    # Cache the unfiltered list so future calls with different form_types
    # don't need to re-fetch
    _filings_cache[cache_key] = unique
    logger.info("CIK %s — %d total unique filings loaded", cik, len(unique))

    # --- Step 6: filter to requested form types ---
    return [f for f in unique if _matches_form_type(f["form"], form_types)]


# Backward-compatible alias so any old callers still work.
get_recent_filings = get_all_filings


# ---------------------------------------------------------------------------
# Filing text download & parse
# ---------------------------------------------------------------------------

def _build_filing_url(cik: str, accession: str, primary_doc: str) -> str:
    """Construct the full URL to a specific filing document."""
    acc_no_dash = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik.lstrip('0')}/{acc_no_dash}/{primary_doc}"
    )


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
                alt_url = (
                    f"https://www.sec.gov{href}"
                    if href.startswith("/")
                    else f"{index_url}{href}"
                )
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

    Uses the full filing history (recent + older archives) so it can find
    filings from several years ago, not just the latest batch.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g. "AAPL").
    target_date : str
        "YYYY-MM-DD" format.
    window_days : int
        Symmetric window size in calendar days.
    form_types : list[str] | None
        Override ``config.SEC_FORM_TYPES``.

    Returns
    -------
    list[dict]
        Filings within the date window, sorted newest-first.
    """
    cik = get_cik(ticker)
    if cik is None:
        logger.warning("CIK not found for %s", ticker)
        return []

    filings = get_all_filings(cik, form_types=form_types)
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
