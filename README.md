# Multimodal Earnings Prediction Engine v1.0

A modular Python 3.12 project that fuses SEC EDGAR filing text (sentiment via
FinBERT, uncertainty/hedging via Loughran-McDonald lexicon) with yfinance price
features (volatility, momentum, gap, volume) to predict 1-day and 3-day
post-earnings returns for large-cap tickers.

## Architecture

```
multimodal_1.0/
├── config.py               # Central settings: tickers, seeds, paths, windows
├── main.py                 # CLI entrypoint (--collect/--features/--train/--evaluate/--all)
├── data/
│   ├── edgar_client.py     # SEC EDGAR: CIK lookup, 8-K/10-Q filing fetch + text parse
│   ├── price_client.py     # yfinance: earnings dates + OHLCV price data
│   └── pipeline.py         # Orchestrates data collection → outputs/raw_data/
├── features/
│   ├── text_features.py    # FinBERT sentiment + Loughran-McDonald lexicon
│   ├── price_features.py   # Momentum, vol, gap, volume spike, liquidity proxy
│   └── fusion.py           # Merge modalities, impute missing values
├── models/
│   ├── train.py            # Chronological split, Logistic + Ridge pipelines
│   └── evaluate.py         # Metrics, confusion matrix, failure analysis
├── plots/
│   └── visualize.py        # All matplotlib/seaborn figures
└── outputs/                # Created at runtime (gitignored)
    ├── raw_data/           #   filings + prices per ticker/date
    ├── features/           #   feature_matrix.csv
    ├── models/             #   joblib-saved sklearn models
    ├── plots/              #   PNG figures
    └── summary.txt         #   written results summary
```

## Prerequisites

- **Python 3.12** (verify with `python3 --version`)
- Internet access for:
  - SEC EDGAR API (earnings filings)
  - Yahoo Finance (price data)
  - Hugging Face Hub (one-time FinBERT model download, ~400 MB)

## Setup

```bash
# 1. Clone / navigate to the project directory
cd multimodal_1.0

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run of `--features` will download the FinBERT model
> (`ProsusAI/finbert`, ~400 MB) from Hugging Face Hub. Subsequent runs use the
> local cache.

## Usage

Run the full pipeline end-to-end:

```bash
python main.py --all
```

Or run individual stages:

```bash
# Stage 1: Collect earnings filings + price data
python main.py --collect

# Stage 2: Extract text + price features, build feature matrix
python main.py --features

# Stage 3: Train direction + magnitude models (1-day & 3-day targets)
python main.py --train

# Stage 4: Evaluate models, generate plots and summary
python main.py --evaluate
```

## Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `TICKERS` | 8 large-caps | AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM |
| `MAX_QUARTERS` | 4 | Earnings events per ticker to fetch |
| `SEED` | 42 | Global random seed for reproducibility |
| `TRAIN_RATIO` | 0.7 | Chronological train/test split ratio |
| `EVENT_WINDOW` | (-20, +10) | Trading days around earnings date |
| `TARGET_HORIZONS` | [1, 3] | Post-earnings return horizons in days |
| `SEC_USER_AGENT` | configurable | Required by SEC EDGAR (set your email) |

## Outputs

After a full run, `outputs/` contains:

| Path | Contents |
|---|---|
| `outputs/raw_data/{TICKER}/{DATE}/` | Filing text (.txt) + price data (.csv) |
| `outputs/features/feature_matrix.csv` | Merged text + price features for all events |
| `outputs/models/` | 4 joblib-saved sklearn pipeline models |
| `outputs/plots/` | Confusion matrices, feature importance, scatter, PnL curves |
| `outputs/summary.txt` | Full written summary: metrics, failures, improvements |

## Module Descriptions

### `data/`
- **edgar_client.py** — HTTP client for SEC EDGAR. Looks up CIK codes, fetches
  8-K and 10-Q filing indexes, downloads and parses filing HTML to clean text.
  Respects SEC rate limit (10 req/sec) and requires a User-Agent header.
- **price_client.py** — Thin wrapper around yfinance. Retrieves earnings dates
  and OHLCV price history for configurable windows around each event.
- **pipeline.py** — Orchestrator that iterates tickers × earnings dates, calls
  both clients, and saves raw data to disk in a structured folder layout.

### `features/`
- **text_features.py** — Loads FinBERT (`ProsusAI/finbert`) for sentiment
  classification (positive/negative/neutral probabilities). Also computes
  lexicon-based features from embedded Loughran-McDonald word lists: uncertainty
  ratio, hedging ratio, forward-guidance strength, litigious ratio. Includes an
  abstract `LLMAdapterInterface` showing how to swap FinBERT for any LLM API.
- **price_features.py** — Computes: pre-earnings momentum (cumulative return),
  pre-earnings realized volatility, post-earnings gap, volume spike ratio,
  Amihud illiquidity proxy, and target returns (1-day, 3-day).
- **fusion.py** — Loads all raw data, calls both feature extractors, merges into
  a single DataFrame. Handles missing values via median imputation + boolean
  flags. Outputs `feature_matrix.csv`.

### `models/`
- **train.py** — Strict chronological split (sort by date, no shuffle). Trains
  4 sklearn pipelines: LogisticRegression (direction) and Ridge (magnitude) for
  each target horizon. All use StandardScaler. Models saved via joblib.
- **evaluate.py** — Classification metrics (accuracy, precision, recall, F1,
  confusion matrix) for direction models. Regression metrics (MAE, RMSE, R²,
  Spearman IC) for magnitude models. Per-ticker breakdown and failure analysis.

### `plots/`
- **visualize.py** — Generates: confusion matrix heatmaps, feature importance
  bar charts, predicted-vs-actual scatter plots, cumulative PnL curves,
  sentiment distribution histograms, per-ticker accuracy bars. All saved as PNG.

## Design Decisions

- **No time-series leakage**: Strict chronological split — all test dates are
  strictly after all training dates. No future information leaks into features.
- **Missing-value handling**: Median imputation for numeric features, boolean
  `text_missing` flag when filing text is unavailable.
- **Reproducible seeds**: numpy, torch, and random seeds set at startup.
- **LLM upgrade path**: `LLMAdapterInterface` in `text_features.py` defines a
  clean interface for swapping FinBERT with any LLM (OpenAI, Anthropic, local).
- **CPU-only**: FinBERT runs on CPU. No GPU required.

## What Worked, What Failed, What to Improve

*(Populated in `outputs/summary.txt` after running the pipeline)*

A full analysis of model performance, feature importance, failure cases, and
recommended next steps is written automatically during evaluation.

## License

Internal prototype — not for distribution.
