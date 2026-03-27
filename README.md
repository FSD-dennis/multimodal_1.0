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

---

## Feature Dictionary

This section provides a complete reference for every feature in the model —
what it measures, how it is computed, and why it matters for predicting
post-earnings price drift at both the **macroeconomic** and
**microstructure** levels.

The project uses **13 input features** across two modalities (text and price)
plus **1 metadata flag**, and produces **2 prediction targets**.

---

### A. Text Features (Extracted from SEC Filings)

These 8 features are derived from 8-K and 10-Q filings downloaded from SEC
EDGAR. They capture **what management says** and **how they say it** — the
narrative signal that complements hard numbers.

#### 1. `sentiment_score`

| | |
|---|---|
| **Definition** | Net sentiment polarity of the filing text. |
| **Computation** | `positive_prob − negative_prob` from FinBERT. Text is chunked into 512-token windows; chunk-level softmax probabilities are averaged to produce a document-level score. |
| **Range** | [−1.0, +1.0]. Positive = optimistic tone; negative = pessimistic. |
| **Micro meaning** | **Firm-level narrative signal.** Management that uses more positive language ("strong growth", "exceeded expectations", "record revenue") is signalling confidence in near-term prospects. Academic research (Loughran & McDonald, 2011; Huang et al., 2014) shows filing sentiment predicts abnormal returns in the 1–5 day window after publication. A high sentiment score suggests the firm's own view of its performance is favorable. |
| **Macro meaning** | **Aggregate earnings tone as an economic barometer.** When sentiment scores trend positive across many firms simultaneously, it reflects broad corporate confidence — often correlated with expansionary GDP, rising PMI, and risk-on market regimes. A sector-wide decline in sentiment can presage margin compression or demand slowdowns before they show up in hard macro data. |
| **Why it matters for post-earnings drift** | Markets under-react to soft information embedded in text. If sentiment is strongly positive but the stock hasn't moved proportionally, there is room for delayed upward drift (the "post-earnings announcement drift" or PEAD anomaly). |

#### 2. `positive_prob`

| | |
|---|---|
| **Definition** | Probability that the filing text is classified as "positive" by FinBERT. |
| **Computation** | Softmax output for the positive class, averaged across all text chunks. |
| **Range** | [0.0, 1.0] |
| **Micro meaning** | Measures the raw intensity of optimistic language. A filing with `positive_prob = 0.85` is overwhelmingly upbeat — management is likely discussing beats, raised guidance, or strong demand. Useful as a standalone signal because it captures one-sided conviction rather than net balance. |
| **Macro meaning** | Tracking `positive_prob` across the S&P 500 gives a "corporate optimism index." Historically, periods where aggregate positive probability exceeds 0.6 align with bull markets and expanding multiples. |

#### 3. `negative_prob`

| | |
|---|---|
| **Definition** | Probability that the filing text is classified as "negative" by FinBERT. |
| **Computation** | Softmax output for the negative class, averaged across all text chunks. |
| **Range** | [0.0, 1.0] |
| **Micro meaning** | Captures the intensity of cautious/pessimistic language: "decline", "headwinds", "lower than expected", "restructuring charges." High values often appear when management is front-running a miss or guiding down. Even when EPS beats, elevated negative tone can signal that the beat was low-quality (e.g., cost cuts, not revenue growth). |
| **Macro meaning** | Spikes in aggregate negative probability across filings tend to lead equity drawdowns by 1–2 quarters. It acts as an early-warning system for earnings recessions — management sees demand softening before it hits GDP prints. |

#### 4. `neutral_prob`

| | |
|---|---|
| **Definition** | Probability that the filing text is classified as "neutral" by FinBERT. |
| **Computation** | Softmax output for the neutral class, averaged across all text chunks. |
| **Range** | [0.0, 1.0] |
| **Micro meaning** | A high neutral probability means the text is **factual and non-committal** — boilerplate risk factors, accounting disclosures, or regulatory language. Very high neutral scores (>0.7) often indicate the filing is dominated by legal/regulatory text rather than management commentary; the informational content for price prediction is low. |
| **Macro meaning** | Rising neutrality across filings can indicate increased regulatory caution or a "wait-and-see" environment where firms avoid forward commitments — typical of late-cycle uncertainty or policy regime changes (rate hike cycles, election years). |
| **Modeling note** | Neutral probability is a useful "quality gate" — events with very high neutral scores may warrant down-weighting in training because the text signal is noisy. |

#### 5. `uncertainty_ratio`

| | |
|---|---|
| **Definition** | Fraction of words in the filing that belong to the Loughran-McDonald uncertainty word list. |
| **Computation** | `count(words ∩ UNCERTAINTY_SET) / total_word_count`. The word list includes: "approximate", "contingent", "could", "doubt", "estimate", "fluctuate", "indefinite", "may", "might", "possible", "predict", "risk", "speculate", "uncertain", "unlikely", "volatile", etc. (~100 terms). |
| **Range** | [0.0, ~0.05] typical; higher = more uncertain language. |
| **Micro meaning** | **Management's own risk assessment.** When a filing is dense with uncertainty words, management is hedging its forward statements. This often appears when: (a) revenue visibility is poor, (b) a product transition is underway, (c) litigation or regulatory risk is elevated, or (d) macro conditions make forecasting difficult. Empirically, high uncertainty filings are followed by higher realized volatility and wider post-earnings price swings. |
| **Macro meaning** | **Leading indicator of implied volatility.** When uncertainty ratios rise across many firms in the same quarter, it signals systemic uncertainty — trade wars, pandemic shocks, rate policy pivots. This cross-sectional signal has been shown to lead VIX by 1–2 months (Loughran & McDonald, 2011). |
| **Why it matters** | High uncertainty → wider distribution of possible outcomes → larger post-earnings moves in either direction. Models can use this to size position conviction or to expect higher absolute returns. |

#### 6. `hedging_ratio`

| | |
|---|---|
| **Definition** | Fraction of words from the Loughran-McDonald hedging/weak-modal word list. |
| **Computation** | `count(words ∩ HEDGING_SET) / total_word_count`. Terms include: "almost", "appear", "apparently", "conceivably", "conditional", "fairly", "generally", "largely", "likelihood", "mostly", "partially", "possibly", "potentially", "presumably", "probably", "relatively", "seemingly", "somewhat", "substantially", "typically", "usually". |
| **Range** | [0.0, ~0.03] typical. |
| **Micro meaning** | **Qualifier density — how much management softens its claims.** A CEO who says "revenue will grow 15%" vs. "revenue will *probably* grow *approximately* 15%" is conveying meaningfully different confidence levels. High hedging ratios indicate management is inserting escape clauses, which frequently precedes guidance misses in subsequent quarters. |
| **Macro meaning** | At the aggregate level, rising hedging language across S&P 500 filings correlates with increasing forecast dispersion among sell-side analysts — a hallmark of uncertain macro environments where the range of outcomes is wide. |
| **Relationship to uncertainty_ratio** | These two features are correlated (~0.4–0.6) but capture different aspects: uncertainty captures **what** is uncertain (risk, volatility, contingencies) while hedging captures **how** management qualifies its own statements. Using both provides finer granularity. |

#### 7. `forward_guidance_strength`

| | |
|---|---|
| **Definition** | Fraction of words signalling forward-looking statements and explicit guidance. |
| **Computation** | `count(words ∩ GUIDANCE_SET) / total_word_count`. Terms include: "anticipate", "expect", "forecast", "guidance", "guide", "intend", "objective", "outlook", "plan", "project", "strategy", "strategic", "target", "will". |
| **Range** | [0.0, ~0.04] typical. |
| **Micro meaning** | **How aggressively management is projecting forward.** A high forward-guidance strength indicates the company is actively setting expectations — issuing targets, reaffirming outlooks, or providing specific forecasts. Firms that give strong guidance typically have better earnings visibility (subscription revenue, long-term contracts). Conversely, firms that suddenly *reduce* guidance language quarter-over-quarter may be signalling deteriorating visibility. |
| **Macro meaning** | **Corporate forward-looking confidence index.** When firms collectively increase guidance language, it suggests the business environment is predictable enough for management to commit publicly — bullish for equities. A drop-off often precedes recessions by 1–2 quarters as firms withdraw specific forward targets. |
| **Alpha signal** | The *change* in forward_guidance_strength from Q(n-1) to Q(n) for the same firm is often more predictive than the absolute level. A sudden increase in guidance language after a period of vagueness can signal an inflection point. |

#### 8. `litigious_ratio`

| | |
|---|---|
| **Definition** | Fraction of words from the Loughran-McDonald litigious word list. |
| **Computation** | `count(words ∩ LITIGIOUS_SET) / total_word_count`. Terms include: "allegation", "arbitration", "claim", "complaint", "court", "defendant", "dispute", "guilty", "indictment", "injunction", "lawsuit", "legal", "legislation", "liability", "litigate", "litigation", "plaintiff", "prosecution", "settlement", "statute", "subpoena", "sue", "trial", "verdict", "violation". |
| **Range** | [0.0, ~0.04] typical. |
| **Micro meaning** | **Legal & regulatory risk exposure.** Filings with elevated litigious language indicate ongoing or anticipated legal proceedings — class-action lawsuits, SEC investigations, patent disputes, antitrust scrutiny. For the stock, this represents a **tail-risk factor**: litigation outcomes are binary (settlement/judgment), hard to price, and can create sudden large moves. High litigious ratios are associated with wider option implied volatility and negative skew. |
| **Macro meaning** | Sector-level litigation spikes often follow regulatory regime changes (new administration, new SEC chair, GDPR-style regulations). Aggregate litigious language in tech filings spiked during the 2020–2024 antitrust wave targeting FAANG companies. |
| **Modeling note** | This feature is most useful as a **risk flag** rather than a directional predictor. It helps the model identify events where the return distribution is fat-tailed and standard momentum/sentiment signals may be overridden by legal outcomes. |

---

### B. Price Features (Engineered from OHLCV Data)

These 5 features are computed from daily Open/High/Low/Close/Volume data
downloaded via yfinance. They capture **what the market is doing** before
and at the moment of the earnings event — the quantitative microstructure
signal.

#### 9. `pre_earnings_momentum`

| | |
|---|---|
| **Definition** | Cumulative total return over the 20 trading days before the earnings announcement. |
| **Computation** | `product(1 + daily_return[t-20 : t-1]) − 1`, where `daily_return = Close[i] / Close[i-1] − 1`. |
| **Range** | Typically [−0.15, +0.20] for large-caps over a 20-day window. |
| **Micro meaning** | **Pre-event price trend — has the market already priced in the news?** A stock that has rallied +15% into earnings has already "pulled forward" good expectations. If the actual report merely *meets* those expectations, the stock may sell off ("buy the rumor, sell the news"). Conversely, a stock that has drifted down −10% into earnings has low expectations baked in, creating an asymmetric setup where even a modest beat triggers a large up-move. |
| **Macro meaning** | When pre-earnings momentum is broadly positive across the market (most stocks trending up into reports), it indicates a **risk-on regime** — investors are front-running earnings optimism, which itself is fueled by strong macro data (employment, consumer spending, PMI). Broad negative momentum into earnings season suggests the market is pricing in an earnings recession. |
| **Academic reference** | Related to the "earnings momentum" factor (Chordia & Shivakumar, 2006) and "price momentum" (Jegadeesh & Titman, 1993). Post-earnings drift is stronger for stocks with momentum aligned with the earnings surprise direction. |

#### 10. `pre_earnings_volatility`

| | |
|---|---|
| **Definition** | Realized volatility (standard deviation of log-returns) over the 20 trading days before earnings. |
| **Computation** | `std(ln(Close[i] / Close[i-1]))` for `i` in `[t-20, t-1]`. |
| **Range** | Typically [0.005, 0.05] daily. Annualized ≈ [8%, 80%]. |
| **Micro meaning** | **Pre-event uncertainty as priced by realized price action.** High pre-earnings vol means the stock has been moving erratically — the market is unsure about the fundamental trajectory. This has two implications: (a) options are likely expensive (high implied vol), and (b) the post-earnings move may be *relatively* smaller because uncertainty is already reflected in pricing. Low pre-earnings vol ("volatility compression") creates the opposite setup — options are cheap, positioning is complacent, and the post-earnings move can be disproportionately large. |
| **Macro meaning** | Average pre-earnings volatility across the market proxies for the realized volatility regime. During periods of low macro volatility (2017 "volmageddon" era, 2024 soft landing), pre-earnings vol compresses across the board, and earnings season becomes the dominant volatility catalyst. During high-vol regimes (2020 COVID, 2022 rate shock), individual earnings events have less incremental impact because the baseline noise level is already high. |
| **Why it matters** | This feature helps the model calibrate **expected move magnitude**. A stock with compressed vol that then surprises will have a larger post-earnings drift than one that was already swinging wildly. |

#### 11. `post_earnings_gap`

| | |
|---|---|
| **Definition** | Overnight gap return between the last pre-earnings close and the earnings-day open. |
| **Computation** | `Open[event_day] / Close[event_day − 1] − 1`. If the previous close is unavailable, falls back to `Close[event_day] / Open[event_day] − 1` (intra-day return). |
| **Range** | Typically [−0.10, +0.15] for large-caps. Can exceed ±20% for high-beta names (TSLA, NVDA). |
| **Micro meaning** | **The market's instantaneous verdict on the earnings report.** This is the "gap" that captures the initial price adjustment as the market digests the headline numbers (EPS beat/miss, revenue, guidance) during after-hours/pre-market trading. The gap is driven by: (a) earnings surprise magnitude relative to consensus, (b) guidance revisions, (c) segment-level beats/misses, and (d) management tone on the call. The gap contains the **most concentrated information** of any feature in this set. |
| **Macro meaning** | The distribution of post-earnings gaps across the market reveals **earnings season quality**. When most gaps are positive and large, the quarter is a strong beat season — this historically coincides with rising analyst estimates, earnings growth re-acceleration, and supportive monetary policy. When gaps skew negative, it signals an earnings recession, often coinciding with tightening financial conditions or demand destruction. |
| **Modeling role** | This feature is available at prediction time because the gap occurs on the event day, before we compute the 1-day and 3-day forward returns (which start from the event-day close). A large positive gap followed by further drift = momentum continuation. A large gap followed by reversal = overreaction. The model learns these continuation/reversal patterns. |
| **Caution** | This is the feature most prone to data-alignment issues. The computation must strictly use the correct trading day as the event date. |

#### 12. `volume_spike`

| | |
|---|---|
| **Definition** | Ratio of earnings-day trading volume to the average daily volume over the prior 20 trading days. |
| **Computation** | `Volume[event_day] / mean(Volume[t-20 : t-1])`. |
| **Range** | Typically [1.5, 8.0] for large-caps. Can exceed 15× for high-impact events. |
| **Micro meaning** | **Intensity of market reaction — how much attention did this event receive?** A volume spike of 5× means five times as many shares changed hands as usual — institutions are actively repositioning. High volume spikes signal **high conviction** in the market's reaction: the price move is likely to sustain (continuation) rather than reverse. A muted volume spike (<2×) with a large gap is more suspicious — the move may be driven by thin liquidity rather than broad institutional flow, making reversal more likely. |
| **Macro meaning** | Aggregate volume spikes across earnings season measure **market engagement**. During risk-off periods, investors may sit out earnings (lower spikes), while during thematic rotations (AI in 2024–2025), specific names see outsized volume spikes as the market reprices sector narratives. |
| **Microstructure insight** | In market microstructure theory, volume is a proxy for **information asymmetry resolution** (Kyle, 1985). The earnings announcement resolves a large information event; the volume spike measures how much disagreement existed beforehand and how many market participants updated their views. Higher volume → more disagreement resolved → more durable price move. |

#### 13. `amihud_illiquidity`

| | |
|---|---|
| **Definition** | Amihud (2002) illiquidity ratio — average of `|daily_return| / dollar_volume` over the pre-earnings window. |
| **Computation** | `mean(|Close[i]/Close[i-1] − 1| / (Close[i] × Volume[i]))` for `i` in `[t-20, t-1]`. |
| **Range** | Varies by orders of magnitude across market-cap tiers. For mega-caps (AAPL, MSFT), values are near 0 (highly liquid). For smaller large-caps, values are higher. |
| **Micro meaning** | **Price impact per unit of trading activity.** A high Amihud ratio means each dollar of volume moves the price more — the stock is illiquid. Around earnings, illiquid stocks tend to gap more (wider bid-ask spreads, thinner order books) and can overshoot, creating larger mean-reversion opportunities. Liquid stocks (low Amihud) absorb news efficiently with less overshoot. |
| **Macro meaning** | **Systemic liquidity indicator.** When Amihud ratios rise across the market, it signals deteriorating market-making capacity — often during risk-off events (VIX spikes), year-end window-dressing periods, or regulatory changes affecting dealer balance sheets. During systemic liquidity crunches (March 2020, SVB week 2023), even mega-cap stocks see their Amihud ratios spike temporarily. |
| **Academic reference** | Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and time-series effects." *Journal of Financial Markets*. The illiquidity premium is one of the best-documented cross-sectional return predictors. |
| **Why it matters** | The model uses this to adjust for **execution reality**: a predicted 2% post-earnings drift in an illiquid name may be larger in magnitude but harder (and more costly) to actually capture in live trading. It also helps distinguish between "real" price moves driven by information and "noise" moves driven by thin markets. |

---

### C. Metadata & Control Features

#### 14. `text_missing`

| | |
|---|---|
| **Definition** | Boolean flag indicating whether filing text was unavailable or too short (<50 words) for this event. |
| **Values** | `True` (text unavailable) or `False` (text features are real). |
| **Why it exists** | Not all earnings events have readily downloadable 8-K or 10-Q filings on SEC EDGAR (especially for recent quarters that may not yet be filed, or for non-US-domiciled companies). When text is missing, the text features (`sentiment_score` through `litigious_ratio`) are imputed with column medians. This flag lets the model **learn to discount imputed text features** — effectively telling it "the text data for this event is synthetic, rely more on price features." |
| **Impact on accuracy** | Events with `text_missing = True` typically have lower prediction accuracy because half the feature space is imputed noise. The model learns this automatically through the boolean flag. |

#### 15. `ticker`

| | |
|---|---|
| **Definition** | Stock ticker symbol (e.g., "AAPL", "NVDA"). |
| **Role** | Metadata only — not used as a model input feature. Used for per-ticker evaluation breakdowns, failure analysis, and stratified reporting. |

#### 16. `event_date`

| | |
|---|---|
| **Definition** | Date of the earnings announcement (YYYY-MM-DD). |
| **Role** | Metadata only — not used as a model input. Used to enforce the **chronological train/test split** (all test dates strictly after all train dates) and for time-series analysis of model performance. |

---

### D. Prediction Targets

#### `ret_1d` (1-Day Post-Earnings Return)

| | |
|---|---|
| **Definition** | Return from the earnings-day close to the next trading day's close. |
| **Computation** | `Close[event_day + 1] / Close[event_day] − 1`. |
| **Why this horizon** | Captures the **immediate continuation or reversal** of the earnings-day gap. Academic PEAD literature (Bernard & Thomas, 1989) shows that a significant portion of post-earnings drift occurs in the first 1–2 trading days as: (a) analyst estimate revisions are published, (b) institutional rebalancing occurs, and (c) retail flow arrives late. |
| **Used for** | Direction model → predicts sign (up/down). Magnitude model → predicts exact return. |

#### `ret_3d` (3-Day Post-Earnings Return)

| | |
|---|---|
| **Definition** | Return from the earnings-day close to the close 3 trading days later. |
| **Computation** | `Close[event_day + 3] / Close[event_day] − 1`. |
| **Why this horizon** | Captures the **medium-term drift** as information diffuses through the market. By day 3, most sell-side analysts have published updated reports, options market makers have adjusted Greeks, and the initial volatility has subsided. The 3-day return separates "efficient" events (where day-1 move fully prices the news) from "drifting" events (where the market continues to adjust). |
| **Used for** | Direction model → predicts sign. Magnitude model → predicts exact return. |

---

### E. Feature Interaction Map

How features relate to each other and the prediction targets:

```
                  ┌─────────────────────────────────────────────┐
                  │            TEXT MODALITY                     │
                  │  sentiment_score ─────────────┐             │
                  │  positive_prob / negative_prob │→ sentiment  │
                  │  neutral_prob ────────────────┘  signal     │
                  │                                              │
                  │  uncertainty_ratio ──┐                       │
                  │  hedging_ratio ──────┤→ confidence / risk    │
                  │  litigious_ratio ────┘  signal               │
                  │                                              │
                  │  forward_guidance_strength → visibility      │
                  └──────────────┬──────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────┐
                  │        FUSION LAYER               │
                  │  (StandardScaler + concatenation) │──→ ret_1d
                  │                                   │──→ ret_3d
                  └──────────────┬────────────────────┘
                                 ▲
                                 │
                  ┌──────────────┴─────────────────────────────┐
                  │            PRICE MODALITY                    │
                  │  pre_earnings_momentum ──→ trend setup       │
                  │  pre_earnings_volatility → vol regime        │
                  │  post_earnings_gap ──────→ initial reaction  │
                  │  volume_spike ───────────→ conviction        │
                  │  amihud_illiquidity ─────→ liquidity context │
                  └─────────────────────────────────────────────┘
```

**Key interactions the model can learn:**
- **High sentiment + large positive gap + high volume** → strong continuation (drift up)
- **High sentiment + large positive gap + low volume** → potential reversal (thin flow)
- **High uncertainty + high pre-vol + large gap** → mean reversion likely (noisy event)
- **Low hedging + strong guidance + positive momentum** → management conviction, drift probable
- **High litigious + high illiquidity** → tail risk, unpredictable — model should have low confidence

---

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


## more to improve
update: create a totally new project for my new understanding of quantitative finance also new coding ability 1. new embedding model 2. better coding structure 3. better dataset understanding what i want more, so many missing data for jpmorgan and nvidia earnings report in manifest.csv my understanding for each features ... 