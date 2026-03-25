"""
Visualisation module — generates all plots and saves as PNG.

Each function takes pre-computed data / metrics and writes a figure to
``outputs/plots/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from models.train import FEATURE_COLS

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Public API — called from main.py
# ---------------------------------------------------------------------------

def generate_all_plots(
    all_metrics: dict[str, dict],
    feature_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Generate every plot and save to *output_dir* (default ``outputs/plots/``)."""
    if output_dir is None:
        output_dir = config.PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, metrics in all_metrics.items():
        if metrics.get("model_type") == "direction":
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                plot_confusion_matrix(
                    cm, title=f"Confusion Matrix — {name}",
                    path=output_dir / f"cm_{name}.png",
                )
        else:
            test_df = metrics.get("test_df")
            preds = metrics.get("predictions")
            target_col = None
            for h in config.TARGET_HORIZONS:
                if f"{h}d" in name:
                    target_col = f"ret_{h}d"
                    break
            if test_df is not None and preds is not None and target_col:
                y_true = test_df[target_col].values
                plot_predicted_vs_actual(
                    y_true, preds,
                    title=f"Predicted vs Actual — {name}",
                    path=output_dir / f"scatter_{name}.png",
                )

    # Feature importance for all models
    for name, metrics in all_metrics.items():
        info = metrics
        # Retrieve pipeline — stored during evaluate
        # We'll get feature importance from the model coefficients
        test_df = info.get("test_df")
        preds = info.get("predictions")
        if test_df is not None:
            _plot_model_feature_importance(
                name, info, output_dir / f"importance_{name}.png",
            )

    # Sentiment distribution
    if not feature_df.empty and "sentiment_score" in feature_df.columns:
        plot_sentiment_distribution(
            feature_df, path=output_dir / "sentiment_distribution.png",
        )

    # Per-ticker accuracy (direction models)
    _plot_per_ticker_accuracy(all_metrics, output_dir / "per_ticker_accuracy.png")

    # Cumulative PnL curve (direction models)
    for name, metrics in all_metrics.items():
        if metrics.get("model_type") == "direction":
            test_df = metrics.get("test_df")
            preds = metrics.get("predictions")
            target_col = None
            for h in config.TARGET_HORIZONS:
                if f"{h}d" in name:
                    target_col = f"ret_{h}d"
                    break
            if test_df is not None and preds is not None and target_col:
                plot_cumulative_returns(
                    test_df, preds, target_col=target_col,
                    title=f"Cumulative PnL — {name}",
                    path=output_dir / f"pnl_{name}.png",
                )

    logger.info("All plots saved → %s", output_dir)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    path: Path | str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Down", "Up"], yticklabels=["Down", "Up"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        logger.info("  → %s", path)
    plt.close(fig)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual Returns",
    path: Path | str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    lims = [
        min(y_true.min(), y_pred.min()) - 0.005,
        max(y_true.max(), y_pred.max()) + 0.005,
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        logger.info("  → %s", path)
    plt.close(fig)


def plot_sentiment_distribution(
    df: pd.DataFrame,
    path: Path | str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Sentiment score histogram
    if "sentiment_score" in df.columns:
        axes[0].hist(df["sentiment_score"].dropna(), bins=20, edgecolor="k", alpha=0.7)
        axes[0].set_xlabel("Sentiment Score (pos − neg)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Sentiment Score Distribution")

    # Uncertainty ratio histogram
    if "uncertainty_ratio" in df.columns:
        axes[1].hist(df["uncertainty_ratio"].dropna(), bins=20, edgecolor="k",
                      alpha=0.7, color="orange")
        axes[1].set_xlabel("Uncertainty Ratio")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Uncertainty Ratio Distribution")

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        logger.info("  → %s", path)
    plt.close(fig)


def plot_cumulative_returns(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    target_col: str = "ret_1d",
    title: str = "Cumulative PnL",
    path: Path | str | None = None,
) -> None:
    """
    Plot cumulative returns of a long/short strategy following model signals.

    Strategy: go long when predicted direction is UP (1), go short when DOWN (0).
    """
    temp = test_df.copy()
    temp = temp.sort_values("event_date").reset_index(drop=True)
    temp["signal"] = predictions
    temp["position"] = temp["signal"].map({1: 1.0, 0: -1.0})
    temp["strategy_ret"] = temp["position"] * temp[target_col]
    temp["cum_strategy"] = (1 + temp["strategy_ret"]).cumprod()
    temp["cum_buyhold"] = (1 + temp[target_col]).cumprod()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(temp)), temp["cum_strategy"].values, label="Model Strategy", linewidth=2)
    ax.plot(range(len(temp)), temp["cum_buyhold"].values, label="Buy & Hold", linewidth=1.5,
            linestyle="--", alpha=0.7)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Event Index (chronological)")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        logger.info("  → %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_model_feature_importance(
    name: str, info: dict, path: Path,
) -> None:
    """Plot feature importance as model coefficient magnitudes."""
    # Try to extract coefficients from the sklearn pipeline
    test_df = info.get("test_df")
    if test_df is None:
        return

    # We need to reload the model to get coefficients
    model_path = config.MODELS_DIR / f"{name}.joblib"
    if not model_path.exists():
        return

    import joblib
    pipe = joblib.load(model_path)
    estimator = pipe.named_steps.get("clf") or pipe.named_steps.get("reg")
    if estimator is None or not hasattr(estimator, "coef_"):
        return

    coefs = estimator.coef_.flatten()
    feature_names = FEATURE_COLS

    if len(coefs) != len(feature_names):
        return

    # Sort by absolute importance
    idx = np.argsort(np.abs(coefs))
    sorted_features = [feature_names[i] for i in idx]
    sorted_coefs = coefs[idx]

    fig, ax = plt.subplots(figsize=(7, max(4, len(sorted_features) * 0.35)))
    colors = ["#d63031" if c < 0 else "#00b894" for c in sorted_coefs]
    ax.barh(sorted_features, sorted_coefs, color=colors, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Coefficient Value")
    ax.set_title(f"Feature Importance — {name}")
    ax.axvline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info("  → %s", path)
    plt.close(fig)


def _plot_per_ticker_accuracy(
    all_metrics: dict[str, dict],
    path: Path,
) -> None:
    """Grouped bar chart of per-ticker accuracy for direction models."""
    from sklearn.metrics import accuracy_score

    records = []
    for name, metrics in all_metrics.items():
        if metrics.get("model_type") != "direction":
            continue
        test_df = metrics.get("test_df")
        preds = metrics.get("predictions")
        if test_df is None or preds is None:
            continue

        target_col = None
        for h in config.TARGET_HORIZONS:
            if f"{h}d" in name:
                target_col = f"ret_{h}d"
                break
        if not target_col:
            continue

        temp = test_df.copy()
        temp["_pred"] = preds
        for ticker, group in temp.groupby("ticker"):
            y_bin = (group[target_col].values >= 0).astype(int)
            y_p = group["_pred"].values
            acc = accuracy_score(y_bin, y_p) if len(y_bin) > 0 else 0
            records.append({"ticker": ticker, "model": name, "accuracy": acc})

    if not records:
        return

    df_plot = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 5))
    models = df_plot["model"].unique()
    n_models = len(models)
    tickers = sorted(df_plot["ticker"].unique())
    x = np.arange(len(tickers))
    width = 0.8 / max(n_models, 1)

    for i, model in enumerate(models):
        sub = df_plot[df_plot["model"] == model]
        accs = [sub[sub["ticker"] == t]["accuracy"].values[0]
                if t in sub["ticker"].values else 0 for t in tickers]
        ax.bar(x + i * width, accs, width, label=model, edgecolor="k", linewidth=0.3)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Ticker Direction Accuracy")
    ax.axhline(0.5, color="red", linewidth=1, linestyle="--", label="Random baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info("  → %s", path)
    plt.close(fig)
