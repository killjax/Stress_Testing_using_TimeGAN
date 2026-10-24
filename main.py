"""main.py

CLI entrypoint for the Stress Testing Trading Strategy Framework with TimeGAN.
Supports training TimeGAN, generating regime-specific synthetic financial data,
computing Discriminative/Predictive/Visualization metrics, and running strategy
stress-testing backtests.
"""

import argparse
import os
import pickle
import numpy as np

from backtesting import evaluate_strategies_across_regimes, plot_stress_test_comparison
from data_loading import (
    DEFAULT_FEATURE_NAMES,
    enforce_financial_consistency,
    label_data,
    real_data_loading,
    real_data_processing,
)
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
from timegan import generate_data_from_saved_models, timegan


def split_by_regime(data_s, data_x, is_generated=False):
    """Splits temporal sequences into (Normal, Crisis, Volatile) subsets."""
    normal_x, crisis_x, volatile_x = [], [], []
    for i, s_vec in enumerate(data_s):
        regime_idx = int(np.argmax(s_vec))
        seq = np.asarray(data_x[i], dtype=np.float32)
        if regime_idx == 0:
            normal_x.append(seq)
        elif regime_idx == 1:
            crisis_x.append(seq)
        else:
            volatile_x.append(seq)
    return normal_x, crisis_x, volatile_x


def normalize_subset(subset, min_val_x, max_val_x):
    """Normalizes a list of 2D sequences using saved MinMaxScaler parameters."""
    return [
        ((np.asarray(seq, dtype=np.float32) - min_val_x) / (max_val_x + 1e-7)).astype(
            np.float32
        )
        for seq in subset
    ]


def evaluate_regime(
    regime_name,
    ori_subset,
    gen_subset,
    min_val_x,
    max_val_x,
    metric_iterations=5,
    disc_iters=2000,
    pred_iters=5000,
    save_plots=False,
):
    """Computes Discriminative and Predictive scores (mean +/- std) for a single regime."""
    if len(ori_subset) == 0 or len(gen_subset) == 0:
        print(f"[{regime_name}] Skipping evaluation (empty subset).")
        return {}

    ori_norm = normalize_subset(ori_subset, min_val_x, max_val_x)
    gen_norm = normalize_subset(gen_subset, min_val_x, max_val_x)

    n_eval = min(len(ori_norm), len(gen_norm))
    rng = np.random.default_rng(42)

    disc_scores = []
    pred_scores = []
    for i in range(metric_iterations):
        idx_o = rng.choice(len(ori_norm), size=n_eval, replace=False)
        idx_g = rng.choice(len(gen_norm), size=n_eval, replace=False)
        sub_o = [ori_norm[j] for j in idx_o]
        sub_g = [gen_norm[j] for j in idx_g]

        d_score = discriminative_score_metrics(
            sub_o, sub_g, iterations=disc_iters, seed=42 + i
        )
        p_score = predictive_score_metrics(
            sub_o, sub_g, iterations=pred_iters, seed=42 + i
        )
        disc_scores.append(d_score)
        pred_scores.append(p_score)

    disc_mean, disc_std = float(np.mean(disc_scores)), float(np.std(disc_scores))
    pred_mean, pred_std = float(np.mean(pred_scores)), float(np.std(pred_scores))

    print(
        f"[{regime_name}] Discriminative Score: {disc_mean:.4f} +/- {disc_std:.4f} | "
        f"Predictive Score: {pred_mean:.4f} +/- {pred_std:.4f}"
    )

    if save_plots:
        visualization(
            ori_norm[:n_eval],
            gen_norm[:n_eval],
            "pca",
            save_path=f"pca_{regime_name.lower()}.png",
        )
        visualization(
            ori_norm[:n_eval],
            gen_norm[:n_eval],
            "tsne",
            save_path=f"tsne_{regime_name.lower()}.png",
        )

    return {
        "regime": regime_name,
        "discriminative_mean": disc_mean,
        "discriminative_std": disc_std,
        "predictive_mean": pred_mean,
        "predictive_std": pred_std,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stress Testing Trading Strategy Framework with TimeGAN"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="AAPL",
        choices=["^GSPC", "AAPL", "GOOG", "MSFT", "AMZN"],
        help="Yahoo Finance ticker symbol",
    )
    parser.add_argument("--start_date", type=str, default="2005-01-01")
    parser.add_argument("--end_date", type=str, default="2024-11-10")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument(
        "--module", type=str, default="gru", choices=["gru", "lstm", "lstmLN"]
    )
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--metric_iterations", type=int, default=5)
    parser.add_argument(
        "--load_saved_pickles",
        action="store_true",
        help="Load existing .pkl datasets instead of downloading/training from scratch",
    )
    parser.add_argument(
        "--load_saved_models",
        action="store_true",
        help="Generate synthetic data from saved_models/ instead of training",
    )
    parser.add_argument(
        "--run_backtest",
        action="store_true",
        default=True,
        help="Run trading strategy stress-testing backtest across regimes",
    )
    args = parser.parse_args()

    feature_names = DEFAULT_FEATURE_NAMES

    if args.load_saved_pickles:
        print("Loading pre-saved regime pickle datasets...")
        with open("ori_data_x_n.pkl", "rb") as f:
            ori_n = enforce_financial_consistency(pickle.load(f), feature_names)
        with open("ori_data_x_c.pkl", "rb") as f:
            ori_c = enforce_financial_consistency(pickle.load(f), feature_names)
        with open("ori_data_x_v.pkl", "rb") as f:
            ori_v = enforce_financial_consistency(pickle.load(f), feature_names)

        with open("generated_data_x_n.pkl", "rb") as f:
            gen_n = enforce_financial_consistency(pickle.load(f), feature_names)
        with open("generated_data_x_c.pkl", "rb") as f:
            gen_c = enforce_financial_consistency(pickle.load(f), feature_names)
        with open("generated_data_x_v.pkl", "rb") as f:
            gen_v = enforce_financial_consistency(pickle.load(f), feature_names)
    else:
        ori_df = real_data_loading(args.data_name, args.start_date, args.end_date)
        feature_names = ori_df.columns.get_level_values(0).tolist()
        ori_data_x = real_data_processing(ori_df, args.seq_len, shuffle=True, seed=42)
        ori_data_s, _ = label_data(ori_data_x, feature_names)

        parameters = {
            "module": args.module,
            "hidden_dim": args.hidden_dim,
            "num_layer": args.num_layers,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "seed": 42,
        }

        if not args.load_saved_models:
            generated_data_s, generated_data_x = timegan(
                ori_data_s, ori_data_x, parameters
            )
        else:
            generated_data_s, generated_data_x = generate_data_from_saved_models(
                ori_data_s, ori_data_x, seed=42
            )

        ori_n, ori_c, ori_v = split_by_regime(ori_data_s, ori_data_x, is_generated=False)
        gen_n, gen_c, gen_v = split_by_regime(
            generated_data_s, generated_data_x, is_generated=True
        )

    scalers = np.load(os.path.join("saved_models", "scalers.npz"))
    min_val_x = scalers["min_val_x"].astype(np.float32)
    max_val_x = scalers["max_val_x"].astype(np.float32)

    if args.metric_iterations > 0:
        print("\n--- Running Model-Centric Evaluation ---")
        evaluate_regime(
            "Normal", ori_n, gen_n, min_val_x, max_val_x, args.metric_iterations
        )
        evaluate_regime(
            "Crisis", ori_c, gen_c, min_val_x, max_val_x, args.metric_iterations
        )
        evaluate_regime(
            "Volatile", ori_v, gen_v, min_val_x, max_val_x, args.metric_iterations
        )

    if args.run_backtest:
        print("\n--- Running Trading Strategy Stress-Testing Backtest ---")
        datasets_dict = {
            ("Real", "Normal"): ori_n,
            ("Synthetic", "Normal"): gen_n,
            ("Real", "Volatile"): ori_v,
            ("Synthetic", "Volatile"): gen_v,
            ("Real", "Crisis"): ori_c,
            ("Synthetic", "Crisis"): gen_c,
        }
        summary_df, detailed_df = evaluate_strategies_across_regimes(
            datasets_dict, feature_names=feature_names
        )
        print("\nStrategy Stress-Testing Summary Table:")
        print(summary_df.to_string(index=False))
        plot_stress_test_comparison(
            detailed_df, save_path="stress_test_results.png", show=False
        )


if __name__ == "__main__":
    main()

