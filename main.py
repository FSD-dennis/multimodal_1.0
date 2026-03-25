#!/usr/bin/env python3
"""
Multimodal Earnings Prediction Engine — CLI entrypoint.

Usage:
    python main.py --all          # Run full pipeline end-to-end
    python main.py --collect      # Stage 1: collect data
    python main.py --features     # Stage 2: extract features
    python main.py --train        # Stage 3: train models
    python main.py --evaluate     # Stage 4: evaluate + plots + summary
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

import numpy as np

import config


def _set_seeds(seed: int = config.SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_collect() -> None:
    """Stage 1: Collect SEC filings + price data for all tickers."""
    from data.pipeline import collect_all

    print("\n" + "=" * 60)
    print("STAGE 1 — DATA COLLECTION")
    print("=" * 60 + "\n")

    manifest = collect_all()
    n_text = sum(1 for r in manifest if r["has_text"] == "True")
    n_price = sum(1 for r in manifest if r["has_prices"] == "True")
    print(f"\nCollection complete: {len(manifest)} events, "
          f"{n_text} with text, {n_price} with prices.")


def run_features() -> None:
    """Stage 2: Extract text + price features → feature matrix."""
    from features.fusion import build_feature_matrix

    print("\n" + "=" * 60)
    print("STAGE 2 — FEATURE EXTRACTION")
    print("=" * 60 + "\n")

    df = build_feature_matrix()
    if df.empty:
        print("ERROR: Feature matrix is empty. Run --collect first.")
        sys.exit(1)
    print(f"\nFeature matrix: {len(df)} rows × {len(df.columns)} columns.")
    print(f"Columns: {list(df.columns)}")


def run_train() -> dict:
    """Stage 3: Train all models. Returns model_results dict."""
    import pandas as pd
    from models.train import train_all_models

    print("\n" + "=" * 60)
    print("STAGE 3 — MODEL TRAINING")
    print("=" * 60 + "\n")

    feature_path = config.FEATURES_DIR / "feature_matrix.csv"
    if not feature_path.exists():
        print("ERROR: Feature matrix not found. Run --features first.")
        sys.exit(1)

    df = pd.read_csv(feature_path)
    results = train_all_models(df)
    print(f"\nTrained {len(results)} models:")
    for name in results:
        print(f"  • {name}")
    return results


def run_evaluate(model_results: dict | None = None) -> None:
    """Stage 4: Evaluate models, generate plots and summary."""
    import pandas as pd
    from models.train import train_all_models
    from models.evaluate import evaluate_all_models
    from plots.visualize import generate_all_plots

    print("\n" + "=" * 60)
    print("STAGE 4 — EVALUATION & PLOTS")
    print("=" * 60 + "\n")

    feature_path = config.FEATURES_DIR / "feature_matrix.csv"
    if not feature_path.exists():
        print("ERROR: Feature matrix not found. Run --features first.")
        sys.exit(1)

    feature_df = pd.read_csv(feature_path)

    # If model_results not passed (standalone --evaluate), re-train
    if model_results is None:
        model_results = train_all_models(feature_df)

    all_metrics = evaluate_all_models(model_results)
    generate_all_plots(all_metrics, feature_df)

    print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
    print(f"  • Summary   : {config.OUTPUT_DIR / 'summary.txt'}")
    print(f"  • Plots     : {config.PLOTS_DIR}")
    print(f"  • Models    : {config.MODELS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Earnings Prediction Engine",
    )
    parser.add_argument("--collect", action="store_true",
                        help="Stage 1: Collect SEC filings + price data")
    parser.add_argument("--features", action="store_true",
                        help="Stage 2: Extract text + price features")
    parser.add_argument("--train", action="store_true",
                        help="Stage 3: Train direction + magnitude models")
    parser.add_argument("--evaluate", action="store_true",
                        help="Stage 4: Evaluate models, generate plots + summary")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages end-to-end")
    args = parser.parse_args()

    if not any([args.collect, args.features, args.train, args.evaluate, args.all]):
        parser.print_help()
        sys.exit(0)

    _setup_logging()
    _set_seeds()
    config.ensure_output_dirs()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Multimodal Earnings Prediction Engine  v1.0           ║")
    print("║   Tickers : " + ", ".join(config.TICKERS).ljust(44) + "║")
    print(f"║   Seed    : {config.SEED:<45}║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.all or args.collect:
        run_collect()

    if args.all or args.features:
        run_features()

    model_results = None
    if args.all or args.train:
        model_results = run_train()

    if args.all or args.evaluate:
        run_evaluate(model_results)

    if args.all:
        print("\n✓ Full pipeline complete.")


if __name__ == "__main__":
    main()
