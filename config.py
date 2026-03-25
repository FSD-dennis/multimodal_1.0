"""
Central configuration for the Multimodal Earnings Prediction Engine.
All tuneable settings live here — no magic numbers elsewhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RAW_DATA_DIR = OUTPUT_DIR / "raw_data"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

# ---------------------------------------------------------------------------
# Universe & scope
# ---------------------------------------------------------------------------
TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "META", "NVDA", "TSLA", "JPM",
]

MAX_QUARTERS: int = 10  # earnings events per ticker to fetch

# Trading-day window around each earnings date  (negative = before)
EVENT_WINDOW: tuple[int, int] = (-20, 10)

# Post-earnings return horizons (in trading days)
TARGET_HORIZONS: list[int] = [1, 3]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.70  # chronological split — first 70 % is train

# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------
# SEC requires a descriptive User-Agent.  Replace with your own name & email.
SEC_USER_AGENT: str = "MultimodalEarningsEngine/1.0 (research@example.com)"
SEC_BASE_URL: str = "https://data.sec.gov"
SEC_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_RATE_LIMIT: float = 0.11  # seconds between requests (≈9 req/s, under 10)

# Filing types to search for earnings-related text
SEC_FORM_TYPES: list[str] = ["8-K", "10-Q"]

# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------
FINBERT_MODEL: str = "ProsusAI/finbert"
FINBERT_MAX_TOKENS: int = 512
FINBERT_BATCH_SIZE: int = 8

# Minimum word count for a filing to be considered usable
MIN_FILING_WORDS: int = 50

# ---------------------------------------------------------------------------
# Feature names (for reference / column ordering)
# ---------------------------------------------------------------------------
TEXT_FEATURES: list[str] = [
    "sentiment_score",
    "positive_prob",
    "negative_prob",
    "neutral_prob",
    "uncertainty_ratio",
    "hedging_ratio",
    "forward_guidance_strength",
    "litigious_ratio",
]

PRICE_FEATURES: list[str] = [
    "pre_earnings_momentum",
    "pre_earnings_volatility",
    "post_earnings_gap",
    "volume_spike",
    "amihud_illiquidity",
]

META_COLUMNS: list[str] = ["ticker", "event_date"]

# ---------------------------------------------------------------------------
# Helper — ensure output directories exist
# ---------------------------------------------------------------------------
def ensure_output_dirs() -> None:
    """Create all output sub-directories if they don't already exist."""
    for d in (RAW_DATA_DIR, FEATURES_DIR, MODELS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
