import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from backtesting import (
    compute_sequence_performance,
    evaluate_strategies_across_regimes,
    plot_stress_test_comparison,
)
from data_loading import (
    DEFAULT_FEATURE_NAMES,
    calculate_window_metrics,
    enforce_financial_consistency,
    label_data,
    real_data_processing,
)
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from timegan import generate_data_from_saved_models, timegan


class TestStressTestingTimeGANPipeline(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.seq_len = 15
        self.dim_x = len(DEFAULT_FEATURE_NAMES)

    def _make_window(self, close_series, log_returns=None):
        T = len(close_series)
        w = np.zeros((T, self.dim_x), dtype=np.float64)
        w[:, 0] = close_series  # Adj Close
        w[:, 1] = close_series  # Close
        w[:, 2] = close_series * 1.01  # High
        w[:, 3] = close_series * 0.99  # Low
        w[:, 4] = close_series * 1.00  # Open
        w[:, 5] = 1e6  # Volume
        if log_returns is None:
            lr = np.zeros(T, dtype=np.float64)
            lr[1:] = np.log(close_series[1:] / close_series[:-1])
        else:
            lr = log_returns
        w[:, 6] = lr  # Log_Return
        w[:, 7] = close_series * 0.02  # ATR
        w[:, 8] = 0.05  # BBW
        w[:, 9] = 0.5  # MACD
        w[:, 10] = 0.3  # MACD_Signal
        w[:, 11] = 50.0  # RSI
        return w

    def test_chronological_order_preserved(self):
        t_steps = 40
        prices = 100.0 * np.exp(np.cumsum(np.full(t_steps, 0.005)))
        raw_w = self._make_window(prices)
        df = pd.DataFrame(raw_w, columns=DEFAULT_FEATURE_NAMES)

        windows = real_data_processing(df, seq_len=15, shuffle=False)
        self.assertEqual(len(windows), t_steps - 15)

        for w in windows:
            close = w[:, 1]
            lr = w[:, 6]
            forward_diff = np.max(np.abs(np.log(close[1:] / close[:-1]) - lr[1:]))
            self.assertLess(forward_diff, 1e-6)

    def test_regime_labeling_forward_mdd(self):
        # 1. Bull rally (+35% steady rise): forward MDD is 0.0 -> must be Normal
        bull_prices = np.linspace(100.0, 135.0, self.seq_len)
        w_bull = self._make_window(bull_prices)

        # 2. Bear crash (-25% drop): forward MDD is 0.25 > 0.20 -> must be Crisis
        crash_prices = np.linspace(100.0, 75.0, self.seq_len)
        w_crash = self._make_window(crash_prices)

        # 3. High volatility sideways (MDD < 0.20, daily vol > 0.0189) -> must be Volatile
        alt_returns = np.array([0.03 if i % 2 == 0 else -0.03 for i in range(self.seq_len)])
        vol_prices = 100.0 * np.exp(np.cumsum(alt_returns))
        w_vol = self._make_window(vol_prices, log_returns=alt_returns)

        ori_data_s, metrics_df = label_data(
            [w_bull, w_crash, w_vol], DEFAULT_FEATURE_NAMES, mode="fixed"
        )
        self.assertEqual(list(metrics_df["label"]), ["Normal", "Crisis", "Volatile"])
        self.assertEqual(ori_data_s[0], [1.0, 0.0, 0.0])
        self.assertEqual(ori_data_s[1], [0.0, 1.0, 0.0])
        self.assertEqual(ori_data_s[2], [0.0, 0.0, 1.0])

    def test_enforce_financial_consistency(self):
        noisy = self.rng.normal(10.0, 20.0, size=(5, self.seq_len, self.dim_x))
        cleaned = enforce_financial_consistency(
            noisy, DEFAULT_FEATURE_NAMES, recompute_indicators=True
        )
        self.assertEqual(len(cleaned), 5)
        for w in cleaned:
            self.assertEqual(w.dtype, np.float32)
            o, h, l, c = w[:, 4], w[:, 2], w[:, 3], w[:, 1]
            self.assertTrue(np.all(h >= np.maximum(o, c) - 1e-5))
            self.assertTrue(np.all(l <= np.minimum(o, c) + 1e-5))
            self.assertTrue(np.all(w[:, 5] >= 0.0))
            self.assertTrue(np.all((w[:, 11] >= 0.0) & (w[:, 11] <= 100.0)))

    def test_timegan_end_to_end_and_conditional_generation(self):
        n_samples = 24
        ori_x = [
            self._make_window(
                100.0 * np.exp(np.cumsum(self.rng.normal(0.0, 0.015, size=self.seq_len)))
            ).astype(np.float32)
            for _ in range(n_samples)
        ]
        ori_s = [
            [1.0, 0.0, 0.0] if i % 3 == 0 else ([0.0, 1.0, 0.0] if i % 3 == 1 else [0.0, 0.0, 1.0])
            for i in range(n_samples)
        ]
        params = {
            "module": "gru",
            "hidden_dim": 8,
            "num_layer": 2,
            "iterations": 3,
            "batch_size": 8,
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_s, gen_x = timegan(ori_s, ori_x, params, save_dir=tmpdir)
            self.assertEqual(len(gen_x), n_samples)
            self.assertEqual(gen_x[0].dtype, np.float32)
            self.assertEqual(gen_x[0].shape, (self.seq_len, self.dim_x))

            # Conditional generation test
            gen_s_c, gen_x_c = generate_data_from_saved_models(
                ori_s,
                ori_x,
                save_dir=tmpdir,
                target_regime="Crisis",
                n_samples=6,
                seed=42,
            )
            self.assertEqual(len(gen_x_c), 6)
            self.assertEqual(gen_x_c[0].dtype, np.float32)

    def test_metrics_and_backtesting(self):
        n_samples = 20
        ori_x = [
            self.rng.uniform(0.1, 0.9, size=(self.seq_len, self.dim_x)).astype(np.float32)
            for _ in range(n_samples)
        ]
        gen_x = [
            self.rng.uniform(0.1, 0.9, size=(self.seq_len, self.dim_x)).astype(np.float32)
            for _ in range(n_samples)
        ]

        d_score = discriminative_score_metrics(ori_x, gen_x, iterations=10, seed=42)
        p_score = predictive_score_metrics(ori_x, gen_x, iterations=10, seed=42)
        self.assertGreaterEqual(d_score, 0.0)
        self.assertLessEqual(d_score, 0.5)
        self.assertGreaterEqual(p_score, 0.0)

        datasets_dict = {
            ("Real", "Normal"): ori_x,
            ("Synthetic", "Normal"): gen_x,
            ("Real", "Crisis"): ori_x[:10],
            ("Synthetic", "Crisis"): gen_x[:10],
        }
        summary_df, detailed_df = evaluate_strategies_across_regimes(
            datasets_dict, feature_names=DEFAULT_FEATURE_NAMES
        )
        self.assertFalse(summary_df.empty)
        self.assertFalse(detailed_df.empty)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig_path = os.path.join(tmpdir, "test_stress.png")
            plot_stress_test_comparison(detailed_df, save_path=fig_path, show=False)
            self.assertTrue(os.path.exists(fig_path))


if __name__ == "__main__":
    unittest.main()

