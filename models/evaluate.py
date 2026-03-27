"""
Model evaluation — metrics, per-ticker breakdown, failure analysis.

Generates classification metrics (accuracy, precision, recall, F1, confusion
matrix) for direction models, and regression metrics (MAE, RMSE, R², Spearman
IC) for magnitude models.  Writes a comprehensive summary to
``outputs/summary.txt``.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

import config
from models.train import FEATURE_COLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_all_models(
    model_results: dict[str, dict],
    output_dir: Path | None = None,
) -> dict[str, dict]:
    """
    Evaluate every trained model and write summary.

    Parameters
    ----------
    model_results : dict
        Output of ``train.train_all_models`` — maps model names to dicts with
        ``pipeline``, ``test_df``, ``target_col``, ``model_type``.
    output_dir : Path
        Where to save ``summary.txt`` (default ``config.OUTPUT_DIR``).

    Returns
    -------
    dict mapping model names to metric dicts (also used by ``visualize``).
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    summary_parts: list[str] = []

    summary_parts.append("=" * 72)
    summary_parts.append("MULTIMODAL EARNINGS PREDICTION ENGINE — EVALUATION SUMMARY")
    summary_parts.append("=" * 72)
    summary_parts.append("")

    for name, info in model_results.items():
        pipe = info["pipeline"]
        test_df: pd.DataFrame = info["test_df"]
        target_col: str = info["target_col"]
        model_type: str = info["model_type"]

        if test_df.empty:
            logger.warning("Empty test set for %s — skipping", name)
            continue

        X_test = test_df[FEATURE_COLS].values
        y_true = test_df[target_col].values

        if model_type == "direction":
            metrics, report = _evaluate_direction(pipe, X_test, y_true, name)
        else:
            metrics, report = _evaluate_magnitude(pipe, X_test, y_true, name)

        # Per-ticker breakdown
        ticker_report = _per_ticker_analysis(pipe, test_df, target_col, model_type)

        # Failure analysis
        failure_report = _failure_analysis(pipe, test_df, target_col, model_type)

        metrics["test_df"] = test_df
        metrics["predictions"] = pipe.predict(X_test)
        all_metrics[name] = metrics

        summary_parts.append(report)
        summary_parts.append(ticker_report)
        summary_parts.append(failure_report)
        summary_parts.append("")

    # --- Model comparison tables ---
    summary_parts.append(_model_comparison_table(all_metrics))

    # --- What-worked / what-failed / improvements ---
    summary_parts.append(_generate_reflections(all_metrics))

    summary_text = "\n".join(summary_parts)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("Summary saved → %s", summary_path)
    print(summary_text)

    return all_metrics


# ---------------------------------------------------------------------------
# Direction (classification) metrics
# ---------------------------------------------------------------------------

def _evaluate_direction(
    pipe, X_test: np.ndarray, y_true: np.ndarray, name: str,
) -> tuple[dict, str]:
    y_binary = (y_true >= 0).astype(int)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_binary, y_pred)
    prec = precision_score(y_binary, y_pred, zero_division=0)
    rec = recall_score(y_binary, y_pred, zero_division=0)
    f1 = f1_score(y_binary, y_pred, zero_division=0)
    cm = confusion_matrix(y_binary, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "model_type": "direction",
    }

    buf = StringIO()
    buf.write(f"--- {name} (Direction / Classification) ---\n")
    buf.write(f"  Accuracy  : {acc:.4f}\n")
    buf.write(f"  Precision : {prec:.4f}\n")
    buf.write(f"  Recall    : {rec:.4f}\n")
    buf.write(f"  F1 Score  : {f1:.4f}\n")
    buf.write(f"  Confusion Matrix:\n")
    buf.write(f"    {cm}\n")
    buf.write(f"\n  Classification Report:\n")
    buf.write(classification_report(y_binary, y_pred, zero_division=0))

    return metrics, buf.getvalue()


# ---------------------------------------------------------------------------
# Magnitude (regression) metrics
# ---------------------------------------------------------------------------

def _evaluate_magnitude(
    pipe, X_test: np.ndarray, y_true: np.ndarray, name: str,
) -> tuple[dict, str]:
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    # Information coefficient (Spearman rank correlation)
    if len(y_true) > 2:
        ic, ic_pval = spearmanr(y_true, y_pred)
    else:
        ic, ic_pval = float("nan"), float("nan")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "ic": float(ic),
        "ic_pval": float(ic_pval),
        "model_type": "magnitude",
    }

    buf = StringIO()
    buf.write(f"--- {name} (Magnitude / Regression) ---\n")
    buf.write(f"  MAE       : {mae:.6f}\n")
    buf.write(f"  RMSE      : {rmse:.6f}\n")
    buf.write(f"  R²        : {r2:.4f}\n")
    buf.write(f"  IC (Spearman) : {ic:.4f}  (p={ic_pval:.4f})\n")

    return metrics, buf.getvalue()


# ---------------------------------------------------------------------------
# Per-ticker breakdown
# ---------------------------------------------------------------------------

def _per_ticker_analysis(
    pipe, test_df: pd.DataFrame, target_col: str, model_type: str,
) -> str:
    buf = StringIO()
    buf.write("  Per-ticker breakdown:\n")

    X = test_df[FEATURE_COLS].values
    y_pred = pipe.predict(X)

    temp = test_df.copy()
    temp["_pred"] = y_pred

    for ticker, group in temp.groupby("ticker"):
        y_t = group[target_col].values
        y_p = group["_pred"].values
        n = len(group)

        if model_type == "direction":
            y_bin = (y_t >= 0).astype(int)
            acc = accuracy_score(y_bin, y_p) if n > 0 else float("nan")
            buf.write(f"    {ticker:6s}: n={n:3d}  accuracy={acc:.3f}\n")
        else:
            mae = mean_absolute_error(y_t, y_p) if n > 0 else float("nan")
            buf.write(f"    {ticker:6s}: n={n:3d}  MAE={mae:.6f}\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def _failure_analysis(
    pipe, test_df: pd.DataFrame, target_col: str, model_type: str,
    top_k: int = 5,
) -> str:
    buf = StringIO()
    buf.write(f"  Failure analysis (top-{top_k} worst predictions):\n")

    X = test_df[FEATURE_COLS].values
    y_true = test_df[target_col].values
    y_pred = pipe.predict(X)

    temp = test_df[["ticker", "event_date", target_col]].copy()
    temp["predicted"] = y_pred

    if model_type == "direction":
        y_bin = (y_true >= 0).astype(int)
        temp["correct"] = y_bin == y_pred
        failures = temp[~temp["correct"]].copy()
        failures["abs_return"] = failures[target_col].abs()
        failures = failures.sort_values("abs_return", ascending=False).head(top_k)
        for _, row in failures.iterrows():
            buf.write(
                f"    {row['ticker']} {row['event_date']}  "
                f"actual_ret={row[target_col]:+.4f}  "
                f"predicted_dir={int(row['predicted'])}\n"
            )
    else:
        temp["error"] = (y_true - y_pred)
        temp["abs_error"] = temp["error"].abs()
        worst = temp.sort_values("abs_error", ascending=False).head(top_k)
        for _, row in worst.iterrows():
            buf.write(
                f"    {row['ticker']} {row['event_date']}  "
                f"actual={row[target_col]:+.6f}  "
                f"predicted={row['predicted']:+.6f}  "
                f"error={row['error']:+.6f}\n"
            )

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def _model_comparison_table(all_metrics: dict[str, dict]) -> str:
    """Build side-by-side comparison tables for direction and magnitude models."""
    buf = StringIO()
    buf.write("=" * 72 + "\n")
    buf.write("MODEL COMPARISON\n")
    buf.write("=" * 72 + "\n\n")

    # --- Direction models ---
    dir_rows: list[tuple[str, float, float, float, float]] = []
    for name, m in all_metrics.items():
        if m.get("model_type") == "direction":
            dir_rows.append((
                name,
                m.get("accuracy", float("nan")),
                m.get("precision", float("nan")),
                m.get("recall", float("nan")),
                m.get("f1", float("nan")),
            ))

    if dir_rows:
        buf.write("Direction models (classification):\n")
        header = f"  {'Model':<40s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}"
        buf.write(header + "\n")
        buf.write("  " + "-" * (len(header) - 2) + "\n")

        # Sort by F1 descending for easy reading
        dir_rows.sort(key=lambda r: r[4], reverse=True)
        best_f1 = dir_rows[0][4]
        for name, acc, prec, rec, f1 in dir_rows:
            marker = " ★" if f1 == best_f1 else ""
            buf.write(
                f"  {name:<40s} {acc:7.4f} {prec:7.4f} {rec:7.4f} {f1:7.4f}{marker}\n"
            )
        buf.write("\n")

    # --- Magnitude models ---
    mag_rows: list[tuple[str, float, float, float, float]] = []
    for name, m in all_metrics.items():
        if m.get("model_type") == "magnitude":
            mag_rows.append((
                name,
                m.get("mae", float("nan")),
                m.get("rmse", float("nan")),
                m.get("r2", float("nan")),
                m.get("ic", float("nan")),
            ))

    if mag_rows:
        buf.write("Magnitude models (regression):\n")
        header = f"  {'Model':<40s} {'MAE':>10s} {'RMSE':>10s} {'R²':>7s} {'IC':>7s}"
        buf.write(header + "\n")
        buf.write("  " + "-" * (len(header) - 2) + "\n")

        # Sort by MAE ascending (lower is better)
        mag_rows.sort(key=lambda r: r[1])
        best_mae = mag_rows[0][1]
        for name, mae, rmse, r2, ic in mag_rows:
            marker = " ★" if mae == best_mae else ""
            buf.write(
                f"  {name:<40s} {mae:10.6f} {rmse:10.6f} {r2:7.4f} {ic:7.4f}{marker}\n"
            )
        buf.write("\n")

    if not dir_rows and not mag_rows:
        buf.write("  No models to compare.\n\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Reflections: what worked, what failed, improvement ideas
# ---------------------------------------------------------------------------

def _generate_reflections(all_metrics: dict[str, dict]) -> str:
    buf = StringIO()
    buf.write("=" * 72 + "\n")
    buf.write("REFLECTIONS: WHAT WORKED, WHAT FAILED, WHAT TO IMPROVE\n")
    buf.write("=" * 72 + "\n\n")

    buf.write("## What Worked\n")
    buf.write("- Pipeline ran end-to-end: data collection → features → modelling → eval.\n")
    buf.write("- Strict chronological split prevents any look-ahead bias.\n")
    buf.write("- FinBERT provides nuanced sentiment beyond simple positive/negative.\n")
    buf.write("- Loughran-McDonald lexicon captures domain-specific language.\n")
    buf.write("- Price features (momentum, vol, gap) are standard alpha signals.\n\n")

    # Check if any direction model beat 50%
    for name, m in all_metrics.items():
        if m.get("model_type") == "direction":
            acc = m.get("accuracy", 0)
            buf.write(f"- {name}: accuracy = {acc:.1%}")
            if acc > 0.5:
                buf.write(" (above random baseline)\n")
            else:
                buf.write(" (at or below random baseline)\n")

    buf.write("\n## What Did Not Work Well\n")
    buf.write("- Small sample size (~32 events for 8 tickers × 4 quarters) limits "
              "model generalisation.\n")
    buf.write("- 8-K filings may not contain rich earnings commentary for all events.\n")
    buf.write("- FinBERT is trained on general financial text; earnings-specific "
              "fine-tuning could help.\n")
    buf.write("- Logistic/Ridge regression cannot capture non-linear feature "
              "interactions.\n")
    buf.write("- Post-earnings drift is notoriously noisy; a single quarter's data "
              "is insufficient.\n\n")

    buf.write("## Recommended Improvements for v2\n")
    buf.write("1. Expand data to 3–5 years of quarterly earnings for 20+ tickers.\n")
    buf.write("2. Add earnings-call transcripts (not just SEC filings) for richer "
              "text signal.\n")
    buf.write("3. Fine-tune FinBERT on earnings-specific labelled data.\n")
    buf.write("4. Try gradient-boosted trees (XGBoost/LightGBM) for non-linear "
              "fusion.\n")
    buf.write("5. Add intraday features (bid-ask spread, order flow imbalance) "
              "for microstructure signal.\n")
    buf.write("6. Implement a transformer-based multimodal fusion architecture "
              "(cross-attention over text + price embeddings).\n")
    buf.write("7. Use an LLM (via the LLMAdapterInterface) for chain-of-thought "
              "reasoning about management tone.\n")
    buf.write("8. Implement walk-forward validation instead of single split.\n")

    return buf.getvalue()
