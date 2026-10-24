# Codebase for "Stress Testing Trading Strategy Framework with TimeGAN Project"

## Project Group Members

- Nishvaan Sai H - 200648
- Rohit Raj - 210874
- Aman Kashyap - 210108
- Jayant Malik - 242110401

## Project Report and Project GitHub Link

- Project Report: [CS787_Project_Report.pdf](CS787_Project_Report.pdf)
- GitHub link: <https://github.com/killjax/Stress_Testing_using_TimeGAN>

## Project Framework

This project provides an end-to-end framework for **Stress Testing Trading Strategies using TimeGAN** with static market regime conditioning (`Normal`, `Crisis`, `Volatile`).

Historical financial time-series data are sourced from the Yahoo Finance API (`yfinance`):

- `^GSPC`: <https://finance.yahoo.com/quote/%5EGSPC/history/?p=^GSPC>
- `AAPL`: <https://finance.yahoo.com/quote/AAPL/history/?p=AAPL>
- `GOOG`: <https://finance.yahoo.com/quote/GOOG/history?p=GOOG>
- `MSFT`: <https://finance.yahoo.com/quote/MSFT/history/?p=MSFT>
- `AMZN`: <https://finance.yahoo.com/quote/AMZN/history/?p=AMZN>

To run the complete pipeline interactively (training, synthetic regime generation, quantitative/qualitative evaluation, and algorithmic trading strategy stress testing), see [`stress_testing_timegan.ipynb`](stress_testing_timegan.ipynb) or run the CLI entrypoint [`main.py`](main.py).

## Reference Paper

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper Link: <https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks>

## Code Structure

### (1) `data_loading.py`

- Loads historical OHLCV financial time-series data and computes 6 technical indicators (`Log_Return`, `ATR`, `BBW`, `MACD`, `MACD_Signal`, `RSI`).
- Slices chronologically ordered time series into overlapping windows (`real_data_processing`).
- Labels windows into `Normal`, `Crisis`, and `Volatile` regimes (`label_data`) based on forward Maximum Drawdown and daily return volatility.
- Enforces financial candlestick and indicator invariants (`enforce_financial_consistency`).

### (2) `timegan.py`

- TensorFlow 2 / Keras implementation of Static-Feature-Integrated TimeGAN (`embedder`, `recovery`, `generator`, `supervisor`, `discriminator`).
- Supports unconditional and regime-conditioned synthetic data generation (`generate_data_from_saved_models` with `target_regime`).

### (3) `metrics/`

- `visualization_metrics.py`: Vectorized PCA and t-SNE analysis comparing Original and Synthetic distributions.
- `discriminative_metrics.py`: Post-hoc GRU classifier measuring `|accuracy - 0.5|` (lower is better; `0.0` is ideal).
- `predictive_metrics.py`: Vectorized Post-hoc GRU next-step predictor (Train-on-Synthetic, Test-on-Real MAE; lower is better).

### (4) `backtesting.py`

- Evaluates algorithmic trading strategies (`Buy & Hold`, `MACD Crossover`, `RSI Mean-Reversion`, `Volatility Target`) across Real and Synthetic market regimes (`Normal`, `Volatile`, `Crisis`).
- Computes Return, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown (`MDD`), Calmar Ratio, 95% Value-at-Risk (`VaR_95`), and 95% Expected Shortfall (`CVaR_95`).

### (5) `utils.py` & `main.py`

- `utils.py`: Vectorized batch/noise generators, train/test splitting, and custom `LayerNormLSTMCell`.
- `main.py`: Command-line interface for training, evaluation, and backtesting.

## Command-Line Usage

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Evaluate and stress-test trading strategies using saved regime datasets
python main.py --load_saved_pickles --metric_iterations 5 --run_backtest

# Full training, generation, evaluation, and strategy stress-testing
python main.py --data_name AAPL --seq_len 60 --module gru --hidden_dim 32 --num_layers 3 --iterations 8000 --batch_size 128 --metric_iterations 5 --run_backtest
```
