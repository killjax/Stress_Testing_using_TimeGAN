import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_FEATURE_NAMES = [
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "Log_Return",
    "ATR",
    "BBW",
    "MACD",
    "MACD_Signal",
    "RSI",
]


def real_data_loading(data_name, start_date, end_date):
    """Load financial time-series data from Yahoo Finance and engineer technical features.

    Args:
        data_name (str): Ticker symbol (e.g., '^GSPC', 'AAPL', 'MSFT', 'GOOG', 'AMZN').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Chronologically ordered DataFrame with 12 features.
    """
    assert data_name in ["^GSPC", "AAPL", "MSFT", "GOOG", "AMZN"]

    ori_data = yf.download(
        data_name, start=start_date, end=end_date, progress=False, auto_adjust=False
    )

    # Force pandas to work on a clean copy and ensure chronological index sorting
    ori_data = ori_data.copy().sort_index(ascending=True)

    # 1. Log Returns
    ori_data["Log_Return"] = np.log(ori_data["Close"] / ori_data["Close"].shift(1))

    # 2. Average True Range (ATR)
    tr1 = ori_data["High"] - ori_data["Low"]
    tr2 = np.abs(ori_data["High"] - ori_data["Close"].shift(1))
    tr3 = np.abs(ori_data["Low"] - ori_data["Close"].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ori_data["ATR"] = true_range.ewm(span=14, adjust=False).mean()

    # 3. Bollinger Band Width (BBW)
    sma_20 = ori_data["Close"].rolling(20).mean()
    std_20 = ori_data["Close"].rolling(20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    ori_data["BBW"] = (upper_band - lower_band) / sma_20

    # 4. & 5. MACD and MACD Signal
    ewm_12 = ori_data["Close"].ewm(span=12, adjust=False).mean()
    ewm_26 = ori_data["Close"].ewm(span=26, adjust=False).mean()
    ori_data["MACD"] = ewm_12 - ewm_26
    ori_data["MACD_Signal"] = ori_data["MACD"].ewm(span=9, adjust=False).mean()

    # 6. RSI (Relative Strength Index)
    delta = ori_data["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss

    # Handle division by zero if avg_loss is 0
    rs[avg_loss == 0] = np.inf
    ori_data["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # Drop all rows with NaN values created by .shift() or rolling windows
    ori_data = ori_data.dropna()

    return ori_data


def real_data_processing(ori_data, seq_len, shuffle=True, seed=None):
    """Slice chronologically ordered data into sequences of length `seq_len`.

    Note: `yfinance` returns data in ascending chronological order (oldest first).
    We preserve this forward chronological order inside each window so that causal
    indicators (Log_Return, MACD, RSI, ATR) and Maximum Drawdown are evaluated
    forward in time.

    Args:
        ori_data (pd.DataFrame or np.ndarray): Raw time-series data of shape (T, D).
        seq_len (int): Window length.
        shuffle (bool): Whether to shuffle the extracted windows (default: True).
        seed (int, optional): Random seed for reproducible window shuffling.

    Returns:
        list[np.ndarray]: List of 2D numpy arrays of shape (seq_len, D).
    """
    ori_data = np.asarray(ori_data, dtype=np.float64)

    temp_data = []
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i : i + seq_len]
        temp_data.append(_x)

    if shuffle and len(temp_data) > 0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(temp_data))
        return [temp_data[i] for i in idx]

    return temp_data


def calculate_window_metrics(window, adj_close_idx=0, log_return_idx=6):
    """Calculates Volatility and Max Drawdown for a single (seq_len, dim) window."""
    window = np.asarray(window, dtype=np.float64)
    prices = np.maximum(window[:, adj_close_idx], 1e-8)
    log_returns = window[:, log_return_idx]

    # 1. Volatility: standard deviation of 1-day log returns
    volatility = float(np.std(log_returns))

    # 2. Maximum Drawdown (MDD) over forward chronological prices
    running_max = np.maximum.accumulate(prices)
    drawdowns = (running_max - prices) / (running_max + 1e-9)
    max_drawdown = float(np.max(drawdowns))

    return volatility, max_drawdown


def label_data(
    ori_data,
    feature_names=None,
    mode="fixed",
    crisis_threshold=0.20,
    volatile_threshold=None,
    crisis_percentile=0.90,
    volatile_percentile=0.85,
):
    """Applies the two-stage market regime labeling process to the dataset.

    Args:
        ori_data (list[np.ndarray]): List of (seq_len, dim) numpy arrays.
        feature_names (list[str], optional): Feature names in column order.
        mode (str): 'fixed' (domain thresholds: MDD > 20%, annualized vol > 30%)
            or 'quantile' (percentile-based thresholds).
        crisis_threshold (float): Fixed MDD threshold for Crisis when mode='fixed'.
        volatile_threshold (float, optional): Fixed daily vol threshold for Volatile.
            Defaults to 0.30 / sqrt(252) (~0.0189).
        crisis_percentile (float): Quantile for Crisis when mode='quantile'.
        volatile_percentile (float): Quantile for Volatile (among non-crisis) when mode='quantile'.

    Returns:
        tuple: (ori_data_s, metrics_df)
            ori_data_s (list[list[float]]): One-hot encoded regime labels:
                Normal:   [1.0, 0.0, 0.0]
                Crisis:   [0.0, 1.0, 0.0]
                Volatile: [0.0, 0.0, 1.0]
            metrics_df (pd.DataFrame): DataFrame with 'volatility', 'mdd', and 'label'.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    adj_close_idx = (
        feature_names.index("Adj Close")
        if "Adj Close" in feature_names
        else feature_names.index("Close")
    )
    log_return_idx = feature_names.index("Log_Return")

    # --- Step 1: Calculate Metrics for Every Window ---
    all_metrics = []
    for window in ori_data:
        vol, mdd = calculate_window_metrics(window, adj_close_idx, log_return_idx)
        all_metrics.append({"volatility": vol, "mdd": mdd})

    metrics_df = pd.DataFrame(all_metrics)

    # --- Step 2: Apply Two-Stage Labeling System ---
    metrics_df["label"] = "Normal"

    if mode == "quantile":
        eff_crisis_thresh = float(metrics_df["mdd"].quantile(crisis_percentile))
        crisis_mask = metrics_df["mdd"] > eff_crisis_thresh
        metrics_df.loc[crisis_mask, "label"] = "Crisis"

        non_crisis_mask = metrics_df["label"] != "Crisis"
        if non_crisis_mask.any():
            eff_vol_thresh = float(
                metrics_df.loc[non_crisis_mask, "volatility"].quantile(
                    volatile_percentile
                )
            )
        else:
            eff_vol_thresh = float(metrics_df["volatility"].quantile(volatile_percentile))
        volatile_mask = non_crisis_mask & (metrics_df["volatility"] > eff_vol_thresh)
        metrics_df.loc[volatile_mask, "label"] = "Volatile"
        print("Labeling complete using quantile thresholds.")
    elif mode == "fixed":
        if volatile_threshold is None:
            # VIX = 30 (0.30 annualized) de-annualized to daily std dev
            volatile_threshold = 0.30 / np.sqrt(252.0)  # ~0.0189

        crisis_mask = metrics_df["mdd"] > crisis_threshold
        metrics_df.loc[crisis_mask, "label"] = "Crisis"

        non_crisis_mask = metrics_df["label"] != "Crisis"
        volatile_mask = non_crisis_mask & (
            metrics_df["volatility"] > volatile_threshold
        )
        metrics_df.loc[volatile_mask, "label"] = "Volatile"
        print("Labeling complete using fixed thresholds.")
    else:
        raise ValueError(f"Unsupported labeling mode: {mode}. Choose 'fixed' or 'quantile'.")

    # --- Final Report ---
    print("\n--- Labeling Results ---")
    print("Final Label Distribution:")
    print(
        metrics_df["label"].value_counts(normalize=True).mul(100).round(1).astype(str)
        + "%"
    )

    # --- Step 3: Create Final One-Hot Encoded List ---
    label_map = {
        "Normal": [1.0, 0.0, 0.0],
        "Crisis": [0.0, 1.0, 0.0],
        "Volatile": [0.0, 0.0, 1.0],
    }

    ori_data_s = [label_map[label] for label in metrics_df["label"]]

    return ori_data_s, metrics_df


def enforce_financial_consistency(
    sequences, feature_names=None, recompute_indicators=False
):
    """Enforces physical/financial invariants on synthetic OHLCV + indicator sequences.

    Specifically guarantees:
      - Positive prices (Open, High, Low, Close, Adj Close > 0)
      - Candlestick ordering: High >= max(Open, Close) and Low <= min(Open, Close)
      - Non-negative Volume, ATR, BBW
      - Bounded RSI in [0, 100]
      - Optionally recomputes technical indicators deterministically from synthetic OHLC.

    Args:
        sequences (list[np.ndarray] or np.ndarray): Synthetic sequences of shape (N, T, D).
        feature_names (list[str], optional): Column names (defaults to DEFAULT_FEATURE_NAMES).
        recompute_indicators (bool): If True, recomputes Log_Return, ATR, BBW, MACD,
            MACD_Signal, and RSI directly from the synthetic OHLC prices.

    Returns:
        list[np.ndarray]: Consistent sequences with float32 dtype.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    idx_map = {name: i for i, name in enumerate(feature_names)}
    cleaned = []

    for seq in sequences:
        w = np.asarray(seq, dtype=np.float64).copy()
        if w.ndim != 2:
            cleaned.append(w.astype(np.float32))
            continue

        # 1. Ensure positive price levels if present
        for price_col in ["Adj Close", "Close", "High", "Low", "Open"]:
            if price_col in idx_map:
                col_i = idx_map[price_col]
                w[:, col_i] = np.maximum(w[:, col_i], 1e-4)

        # 2. Enforce candlestick invariants: High >= max(Open, Close, Low) & Low <= min(Open, Close, High)
        if all(k in idx_map for k in ["Open", "High", "Low", "Close"]):
            o_i, h_i, l_i, c_i = (
                idx_map["Open"],
                idx_map["High"],
                idx_map["Low"],
                idx_map["Close"],
            )
            o, h, l, c = w[:, o_i], w[:, h_i], w[:, l_i], w[:, c_i]
            new_high = np.maximum.reduce([h, o, c, l])
            new_low = np.minimum.reduce([l, o, c, h])
            w[:, h_i] = new_high
            w[:, l_i] = new_low

        # 3. Enforce non-negative volume & volatility indicators, bounded RSI
        for non_neg_col in ["Volume", "ATR", "BBW"]:
            if non_neg_col in idx_map:
                c_idx = idx_map[non_neg_col]
                w[:, c_idx] = np.maximum(w[:, c_idx], 0.0)

        if "RSI" in idx_map:
            r_idx = idx_map["RSI"]
            w[:, r_idx] = np.clip(w[:, r_idx], 0.0, 100.0)

        # 4. Optional deterministic indicator recomputation from synthetic OHLC
        if recompute_indicators and all(
            k in idx_map for k in ["Close", "High", "Low"]
        ):
            s_close = pd.Series(w[:, idx_map["Close"]])
            s_high = pd.Series(w[:, idx_map["High"]])
            s_low = pd.Series(w[:, idx_map["Low"]])

            if "Log_Return" in idx_map and len(w) > 1:
                lr_idx = idx_map["Log_Return"]
                w[1:, lr_idx] = np.log(s_close.iloc[1:].values / s_close.iloc[:-1].values)
                w[0, lr_idx] = np.clip(w[0, lr_idx], -0.25, 0.25)

            if "ATR" in idx_map:
                tr1 = s_high - s_low
                tr2 = (s_high - s_close.shift(1).bfill()).abs()
                tr3 = (s_low - s_close.shift(1).bfill()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                w[:, idx_map["ATR"]] = tr.ewm(span=14, adjust=False).mean().values

            if "BBW" in idx_map:
                sma_20 = s_close.rolling(20, min_periods=2).mean().bfill()
                std_20 = s_close.rolling(20, min_periods=2).std().bfill().fillna(0.0)
                w[:, idx_map["BBW"]] = ((4.0 * std_20) / (sma_20 + 1e-8)).values

            if "MACD" in idx_map:
                ewm_12 = s_close.ewm(span=12, adjust=False).mean()
                ewm_26 = s_close.ewm(span=26, adjust=False).mean()
                macd = ewm_12 - ewm_26
                w[:, idx_map["MACD"]] = macd.values
                if "MACD_Signal" in idx_map:
                    w[:, idx_map["MACD_Signal"]] = (
                        macd.ewm(span=9, adjust=False).mean().values
                    )

            if "RSI" in idx_map:
                delta = s_close.diff(1).fillna(0.0)
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.ewm(span=14, adjust=False).mean()
                avg_loss = loss.ewm(span=14, adjust=False).mean()
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                w[:, idx_map["RSI"]] = np.clip(rsi.values, 0.0, 100.0)

        cleaned.append(w.astype(np.float32))

    return cleaned
