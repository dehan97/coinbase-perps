import logging
import itertools
import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from config.configuration import Config
from helpers.trademgmt.signals_and_filters import (
    SignalCache,
    generate_signals,
    load_data_for_timeframe,
)
from helpers.trademgmt.trade_simulation import (
    TradeCache,
    calculate_trade_outcomes,
    fill_trades_cache,
)

# Setup logging
logging.basicConfig(
    filename=f"{Config().logs_path}/backtest.log",
    level=logging.INFO,
    format=Config().logging_format,
)

# Configuration for backtesting
timeframes = Config().time_intervals
signal_methods = Config().signal_methods
filter_methods = Config().filter_methods
holding_periods = Config().holding_periods
tp_sl_values = Config().tp_sl_values
zscore_lookback_values = Config().zscore_lookback_values
use_absolute_returns_options = Config().use_absolute_returns_options

# Strategy mapping file
strategy_mapping_file = f"{Config().results_path}/strategy_mapping.json"

# Load existing strategy mapping if it exists
if os.path.exists(strategy_mapping_file):
    with open(strategy_mapping_file, "r") as f:
        strategy_mapping = json.load(f)
else:
    strategy_mapping = {}


def save_strategy_mapping():
    """Save the strategy mapping to the file."""
    with open(strategy_mapping_file, "w") as f:
        json.dump(strategy_mapping, f, indent=4)


def max_dd(returns):
    """Assumes returns is a pandas Series"""
    returns = returns.astype(np.float64)  # Ensure high precision
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    returns = np.clip(returns, -0.99, 1e6)  # Ensure valid range for log1p
    log_r = np.log1p(returns)
    log_cumsum = log_r.cumsum()
    log_cumsum = np.clip(log_cumsum, -700, 50)  # Prevents extreme values
    r = np.exp(log_cumsum)

    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.idxmin()
    start = r.loc[:end].idxmax()
    return mdd, start, end


def calculate_strategy_metrics(trades_file: str):
    """Calculates extensive strategy performance metrics per symbol and overall, incorporating trade costs."""
    trades = pd.read_parquet(trades_file)

    if trades.empty:
        return None

    # Compute trade returns based on signal
    trades["trade_return"] = trades["signal"] * trades["final_return_long"]

    # Apply trade costs: 0.7% per trade (one-way)
    trade_costs = (
        2 * Config().one_way_trade_costs * abs(trades["signal"])
    )  # Absolute trade size to account for both buy/sell trades
    trades["trade_return"] -= trade_costs  # Deducting cost from returns

    # Drop NaN values
    trades.dropna(subset=["trade_return"], inplace=True)

    # Calculate daily returns per symbol
    daily_returns = (
        trades.groupby(["start", "symbol"], observed=False)["trade_return"]
        .sum()
        .unstack(fill_value=0)
    )

    # Calculate equal-weighted strategy daily returns (1/n allocation)
    daily_returns["ALL"] = daily_returns.mean(axis=1)

    def compute_metrics(returns, symbol):
        """Helper function to compute extensive strategy metrics."""
        num_days = len(returns)
        annual_factor = 252  # Number of trading days in a year

        # Categorize returns into positive, negative, and no return
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        zero_returns = returns[returns == 0]

        def aggregate_metrics(subset_returns, category):
            """Computes all relevant metrics for a specific subset of trades (overall, positive, negative, zero)."""
            if subset_returns.empty:
                return {
                    f"{category} Sharpe Ratio": None,
                    f"{category} Sortino Ratio": None,
                    f"{category} Win Rate": None,
                    f"{category} Avg Trade Return": None,
                    f"{category} Total Return": None,
                    f"{category} Annualized Return": None,
                    f"{category} Volatility": None,
                    f"{category} Max Drawdown": None,
                    f"{category} Calmar Ratio": None,
                    f"{category} Profit Factor": None,
                    f"{category} Expectancy": None,
                }

            std_dev = np.nan_to_num(
                subset_returns.std(), nan=1e-10, posinf=1e10, neginf=-1e10
            )

            # Avoid division by zero or NaN
            if std_dev == 0 or np.isnan(std_dev):
                sharpe_ratio = 0  # Set Sharpe Ratio to 0 if volatility is zero
            else:
                sharpe_ratio = subset_returns.mean() / std_dev * (annual_factor**0.5)

            downside_returns = subset_returns[subset_returns < 0]
            sortino_ratio = (
                subset_returns.mean() / downside_returns.std() * (annual_factor**0.5)
                if not downside_returns.empty and downside_returns.std() != 0
                else 0.0
            )

            # 1️⃣ Replace infinities with NaN, then fill NaNs with 0
            subset_returns = subset_returns.replace([np.inf, -np.inf], np.nan).fillna(0)

            # 2️⃣ Clip extreme values to prevent log1p issues
            subset_returns = np.clip(
                subset_returns, -0.99, 1e6
            )  # ✅ Ensures valid log1p range

            # 3️⃣ Apply log1p safely
            log_sum = np.log1p(subset_returns).sum()

            # 4️⃣ Prevent extreme log_sum values before exponentiation
            log_sum = np.clip(log_sum, -700, 50)  # ✅ Avoids np.exp() overflow

            # 5️⃣ Use numerically stable exponentiation
            total_return = np.expm1(log_sum)  # ✅ Handles small values more accurately

            annualized_return = (
                (1 + total_return) ** (annual_factor / num_days) - 1
                if num_days > 0
                else None
            )
            volatility = (
                np.nan_to_num(subset_returns.std(), nan=0, posinf=1e10, neginf=-1e10)
                * (annual_factor**0.5)
                if not subset_returns.empty
                else None
            )

            mdd, _, _ = max_dd(subset_returns)
            calmar_ratio = annualized_return / abs(mdd) if mdd != 0 else None
            profit_factor = (
                (
                    subset_returns[subset_returns > 0].sum()
                    / abs(subset_returns[subset_returns < 0].sum())
                )
                if subset_returns[subset_returns < 0].sum() != 0
                else None
            )
            expectancy = subset_returns.mean()

            return {
                f"{category} Sharpe Ratio": sharpe_ratio,
                f"{category} Sortino Ratio": sortino_ratio,
                f"{category} Win Rate": (
                    (subset_returns > 0).sum() / len(subset_returns)
                    if len(subset_returns) > 0
                    else None
                ),
                f"{category} Avg Trade Return": subset_returns.mean(),
                f"{category} Total Return": total_return,
                f"{category} Annualized Return": annualized_return,
                f"{category} Volatility": volatility,
                f"{category} Max Drawdown": mdd,
                f"{category} Calmar Ratio": calmar_ratio,
                f"{category} Profit Factor": profit_factor,
                f"{category} Expectancy": expectancy,
            }

        # Aggregate metrics for each category
        overall_metrics = aggregate_metrics(returns, "Overall")
        positive_metrics = aggregate_metrics(positive_returns, "Positive")
        negative_metrics = aggregate_metrics(negative_returns, "Negative")
        zero_metrics = aggregate_metrics(zero_returns, "Zero")

        # Additional execution-based metrics
        num_trades = trades[trades["symbol"] == symbol]["signal"].ne(0).sum()
        turnover = trades[trades["symbol"] == symbol]["signal"].diff().abs().sum()
        exposure_time = num_trades / num_days if num_days > 0 else None

        # Consecutive Wins & Losses
        win_streaks = (
            (returns > 0).astype(int).groupby((returns <= 0).astype(int).cumsum()).sum()
        )
        loss_streaks = (
            (returns < 0).astype(int).groupby((returns >= 0).astype(int).cumsum()).sum()
        )

        max_consec_wins = win_streaks.max() if not win_streaks.empty else 0
        max_consec_losses = loss_streaks.max() if not loss_streaks.empty else 0

        return {
            **overall_metrics,
            **positive_metrics,
            **negative_metrics,
            **zero_metrics,
            "Number of Trades": num_trades,
            "Turnover": turnover,
            "Exposure Time": exposure_time,
            "Max Consecutive Wins": max_consec_wins,
            "Max Consecutive Losses": max_consec_losses,
            "Symbol": symbol,
        }

    # Compute metrics per symbol (including 'ALL')
    symbol_metrics = []
    for symbol in daily_returns.columns:
        metrics = compute_metrics(daily_returns[symbol], symbol)
        symbol_metrics.append(metrics)

    return pd.DataFrame(symbol_metrics)


def save_strategy_metrics(
    strategy_dir: str, strategy_config: dict, metrics_df: pd.DataFrame
):
    """Saves strategy metrics and configuration to a Parquet file."""
    metrics_file = f"{strategy_dir}/metrics.parquet"

    # Extract signal and filter parameters into separate columns
    strategy_config_flat = strategy_config.copy()

    signal_params = strategy_config_flat.pop("signal_params", {})
    filter_params = strategy_config_flat.pop("filter_params", {})

    strategy_config_flat["Signal Parameters"] = [signal_params] * len(metrics_df)
    strategy_config_flat["Filter Parameters"] = [filter_params] * len(metrics_df)

    # Merge strategy config with metrics
    for key, value in strategy_config_flat.items():
        metrics_df[key] = value

    # Save as Parquet
    metrics_df.to_parquet(metrics_file, index=False)


def run_backtest():
    # Initialize caches
    signal_cache = SignalCache()
    trade_cache = TradeCache()

    strategy_id = len(strategy_mapping) + 1  # Start from the next strategy

    try:
        filter_combinations = [
            (method, params)
            for method, param_list in Config().filter_methods.items()
            for params in param_list
        ]

        # Iterate over all possible backtesting configurations
        total_combinations = list(
            itertools.product(
                timeframes,
                signal_methods.keys(),
                filter_combinations,
                holding_periods,
                tp_sl_values,
                zscore_lookback_values,
                use_absolute_returns_options,
            )
        )

        # Iterate over all combinations with progress bar
        for combination in tqdm(
            total_combinations, desc="Backtesting Progress", unit="strategy"
        ):
            (
                timeframe,
                signal_method,
                (filter_method, filter_params),
                holding_period,
                tp_sl,
                zscore_lookback,
                use_absolute_returns,
            ) = combination

            for signal_params in signal_methods[signal_method]:
                # Create a unique identifier for the strategy
                strategy_label = f"strategy_{strategy_id}"
                strategy_config = {
                    "timeframe": timeframe,
                    "signal_method": signal_method,
                    "signal_params": signal_params,
                    "filter_method": filter_method,
                    "filter_params": filter_params,
                    "holding_period": holding_period,
                    "tp_sl": tp_sl,
                    "zscore_lookback": zscore_lookback,
                    "use_absolute_returns": use_absolute_returns,
                }

                # Define strategy directory and file paths
                strategy_dir = f"{Config().results_path}/{strategy_label}"
                trades_file = f"{strategy_dir}/trades.parquet"
                metrics_file = f"{strategy_dir}/metrics.parquet"

                # Skip processing if results already exist
                if os.path.exists(trades_file):
                    logging.info(f"Skipping {strategy_label}, results already exist.")
                    print(f"Skipping {strategy_label}, results already exist.")

                    # Compute and save metrics if not already done
                    if not os.path.exists(metrics_file):
                        metrics = calculate_strategy_metrics(trades_file)
                        save_strategy_metrics(strategy_dir, strategy_config, metrics)
                        print(f"Saved metrics for {strategy_label}.")
                    else:
                        print(f"Metrics for {strategy_label} already exist.")

                    strategy_id += 1
                    continue

                logging.info(f"Starting {strategy_label}: {strategy_config}")
                print(f"Processing {strategy_label}: {strategy_config}")

                # Generate signals
                signals = generate_signals(
                    timeframe=timeframe,
                    signal_method=signal_method,
                    signal_params=signal_params,
                    filter_method=filter_method,
                    filter_params=filter_params,
                    cache=signal_cache,
                )
                logging.info(f"{strategy_label}: Generated {len(signals)} signals.")
                print(f"{strategy_label}: {len(signals)} signals generated.")
                # print(signals["signal"].value_counts()) #signals contains the actionable signals only.

                # print(f"{signals.shape}")
                # print("Null values before merge:")
                # print(signals.isnull().sum())

                # Merge with price data to get back all rows.
                signals = signals[["symbol", "start", "signal"]].merge(
                    load_data_for_timeframe(timeframe).reset_index()[
                        ["symbol", "start", "close", "returns", "volume"]
                    ],
                    how="right",
                    on=["symbol", "start"],
                )

                # print(f"{signals.shape}")
                # print("Null values after merge:")
                # print(signals.isnull().sum())

                # Fill missing signals and returns with 0
                signals["signal"] = signals["signal"].fillna(0)
                signals["returns"] = signals["returns"].fillna(0)

                # print(f"{signals.columns=}")
                # Calculate trade outcomes
                trade_results = calculate_trade_outcomes(
                    df=signals,
                    holding_period=holding_period,
                    tp_sl=tp_sl,
                    zscore_lookback=zscore_lookback,
                    use_absolute_returns=use_absolute_returns,
                    cache=trade_cache,
                    timeframe=timeframe,
                )

                logging.info(
                    f"{strategy_label}: Calculated outcomes for {len(trade_results)} trades."
                )
                print(
                    f"{strategy_label}: Outcomes calculated for {len(trade_results)} trades."
                )
                # print(f"{trade_results['signal'].value_counts()=}")

                # Create strategy directory
                os.makedirs(strategy_dir, exist_ok=True)

                # Save trade results
                trade_results[trade_results["signal"] != 0].to_parquet(
                    trades_file, index=False
                )

                logging.info(f"{strategy_label}: Saved results to {trades_file}.")
                print(f"Results for {strategy_label} saved to {trades_file}")

                # Compute and save strategy metrics
                metrics = calculate_strategy_metrics(trades_file)
                save_strategy_metrics(strategy_dir, strategy_config, metrics)
                print(f"Saved metrics for {strategy_label}.")

                # Add strategy to mapping and save
                strategy_mapping[strategy_label] = strategy_config
                save_strategy_mapping()

                strategy_id += 1
    finally:
        # Close caches
        signal_cache.close()
        logging.info("Caches closed. Backtesting completed.")
        print("Backtesting completed.")


if __name__ == "__main__":
    # RUN CACHE FILL
    # Initialize caches
    trade_cache = TradeCache()
    signal_cache = SignalCache()

    # Define parameter ranges
    holding_periods = Config().holding_periods
    tp_sl_values = Config().tp_sl_values
    zscore_lookback_values = Config().zscore_lookback_values

    for timeframe in Config().time_intervals:
        # Load market data
        df_prices = load_data_for_timeframe(timeframe).reset_index()[
            ["symbol", "start", "close", "returns"]
        ]

        # Precompute & store all trade results
        fill_trades_cache(
            df=df_prices,
            timeframe=timeframe,
            holding_periods=holding_periods,
            tp_sl_values=tp_sl_values,
            zscore_lookback_values=zscore_lookback_values,
            cache=trade_cache,
        )

    print("Trade cache is now fully populated!")

    # RUN BACKTESTING
    print("Starting backtesting process...")
    logging.info("Backtesting process initiated.")
    run_backtest()
    print("Backtesting process completed.")
    logging.info("Backtesting process completed.")
