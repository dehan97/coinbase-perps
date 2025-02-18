import logging
import itertools
import os
import json
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
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

import warnings

warnings.filterwarnings("ignore")

# Configuration for backtesting
timeframes = Config().time_intervals
signal_methods = Config().signal_methods
filter_methods = Config().filter_methods
holding_periods = Config().holding_periods
tp_sl_values = Config().tp_sl_values
zscore_lookback_values = Config().zscore_lookback_values

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


def calculate_strategy_metrics(trades_file: str) -> pd.DataFrame:
    """
    Calculates extensive strategy performance metrics per symbol and overall, incorporating trade costs.
    If no trades or daily returns are found, returns an empty DataFrame (rather than None).
    """
    trades = pd.read_parquet(trades_file)
    if trades.empty:
        print(f"No trades found in {trades_file}, skipping metrics.")
        # Return an empty DataFrame with appropriate columns (optional)
        return pd.DataFrame()

    # 1) Compute trade returns (including cost)
    trades["trade_return"] = trades["signal"] * trades["final_return_long"]

    # Apply 2 * one-way costs to each position
    trade_costs = 2 * Config().one_way_trade_costs
    trades["trade_return"] -= trade_costs

    # Drop any leftover NaNs
    trades.dropna(subset=["trade_return"], inplace=True)

    # 2) Create subsets for long/short/recent
    long_trades = trades[trades["signal"] == 1]
    short_trades = trades[trades["signal"] == -1]
    recent_trades = trades[
        trades["start"]
        >= (pd.to_datetime(Config().end_date_range) - pd.DateOffset(years=1))
    ]

    # 3) Aggregate daily returns by symbol
    daily_returns = (
        trades.groupby(["start", "symbol"], observed=False)["trade_return"]
        .sum()
        .unstack(fill_value=0)
    )
    buy_daily_returns = (
        long_trades.groupby(["start", "symbol"])["trade_return"]
        .sum()
        .unstack(fill_value=0)
    )
    sell_daily_returns = (
        short_trades.groupby(["start", "symbol"])["trade_return"]
        .sum()
        .unstack(fill_value=0)
    )
    recent_daily_returns = (
        recent_trades.groupby(["start", "symbol"])["trade_return"]
        .sum()
        .unstack(fill_value=0)
    )

    # If no daily returns at all, skip
    if daily_returns.empty:
        print(f"No daily returns from {trades_file}, skipping metrics.")
        return pd.DataFrame()

    # 4) Helper to compute sub-metrics for each category
    def aggregate_metrics(subset_returns: pd.Series, category: str) -> dict:
        """Generic aggregator for one subset (Overall / Long / Short / Recent)."""
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

        annual_factor = 252
        std_dev = float(np.nan_to_num(subset_returns.std(), nan=1e-10))
        mean_return = subset_returns.mean()

        # Sharpe
        sharpe_ratio = (
            0.0 if std_dev <= 1e-10 else mean_return / std_dev * (annual_factor**0.5)
        )

        # Sortino
        downside = subset_returns[subset_returns < 0]
        if not downside.empty and downside.std() != 0:
            sortino_ratio = mean_return / downside.std() * (annual_factor**0.5)
        else:
            sortino_ratio = 0.0

        # Clean infinite / NaNs
        subset_returns = subset_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        subset_returns = np.clip(subset_returns, -0.99, 1e6)

        # Total return via log1p
        log_sum = np.log1p(subset_returns).sum()
        log_sum = np.clip(log_sum, -700, 50)
        total_return = np.expm1(log_sum)

        # Annualized return
        n_days = len(subset_returns)
        annualized_return = None
        if n_days > 0:
            annualized_return = (1.0 + total_return) ** (annual_factor / n_days) - 1.0

        volatility = None
        if not subset_returns.empty:
            volatility = subset_returns.std() * (annual_factor**0.5)

        # Max Drawdown
        mdd, _, _ = max_dd(subset_returns)
        calmar_ratio = (
            annualized_return / abs(mdd) if (mdd != 0 and annualized_return) else None
        )

        # Profit Factor
        neg_sum = subset_returns[subset_returns < 0].sum()
        pf = None
        if neg_sum != 0:
            pf = subset_returns[subset_returns > 0].sum() / abs(neg_sum)

        # Expectancy
        expectancy = mean_return

        # Win Rate
        win_rate = None
        if len(subset_returns) > 0:
            win_rate = (subset_returns > 0).sum() / len(subset_returns)

        return {
            f"{category} Sharpe Ratio": sharpe_ratio,
            f"{category} Sortino Ratio": sortino_ratio,
            f"{category} Win Rate": win_rate,
            f"{category} Avg Trade Return": mean_return,
            f"{category} Total Return": total_return,
            f"{category} Annualized Return": annualized_return,
            f"{category} Volatility": volatility,
            f"{category} Max Drawdown": mdd,
            f"{category} Calmar Ratio": calmar_ratio,
            f"{category} Profit Factor": pf,
            f"{category} Expectancy": expectancy,
        }

    def compute_symbol_metrics(symbol: str) -> dict:
        """
        For a single symbol, compute Overall/Long/Short/Recent.
        """
        overall_series = daily_returns[symbol]
        long_series = buy_daily_returns.get(symbol, pd.Series(dtype=float))
        short_series = sell_daily_returns.get(symbol, pd.Series(dtype=float))
        recent_series = recent_daily_returns.get(symbol, pd.Series(dtype=float))

        # Overall, Long, Short, Recent
        overall = aggregate_metrics(overall_series, "Overall")
        long_met = aggregate_metrics(long_series, "Long")
        short_met = aggregate_metrics(short_series, "Short")
        recent_met = aggregate_metrics(recent_series, "Recent")

        # Additional execution-based metrics
        symbol_trades = trades[trades["symbol"] == symbol]
        num_days = len(overall_series)

        num_trades = symbol_trades["signal"].ne(0).sum()
        num_long_trades = symbol_trades[symbol_trades["signal"] == 1]["signal"].count()
        num_short_trades = symbol_trades[symbol_trades["signal"] == -1][
            "signal"
        ].count()
        turnover = symbol_trades["signal"].diff().abs().sum()
        exposure_time = num_trades / num_days if num_days > 0 else None

        # consecutive wins/losses
        win_streaks = (
            (overall_series > 0)
            .astype(int)
            .groupby((overall_series <= 0).astype(int).cumsum())
            .sum()
        )
        loss_streaks = (
            (overall_series < 0)
            .astype(int)
            .groupby((overall_series >= 0).astype(int).cumsum())
            .sum()
        )

        max_consec_wins = win_streaks.max() if not win_streaks.empty else 0
        max_consec_losses = loss_streaks.max() if not loss_streaks.empty else 0

        # Merge all metrics
        result = {
            "Symbol": symbol,
            **overall,
            **long_met,
            **short_met,
            **recent_met,
            "Num Trades": num_trades,
            "Num Long Trades": num_long_trades,
            "Num Short Trades": num_short_trades,
            "Turnover": turnover,
            "Exposure Time": exposure_time,
            "Max Consecutive Wins": max_consec_wins,
            "Max Consecutive Losses": max_consec_losses,
        }
        return result

    # 5) Build a list of metrics for each symbol
    symbol_metrics = []
    for symbol in daily_returns.columns:
        symbol_metrics.append(compute_symbol_metrics(symbol))

    # 6) Return a DataFrame
    if not symbol_metrics:
        return pd.DataFrame()
    return pd.DataFrame(symbol_metrics)


def save_strategy_metrics(
    strategy_dir: str, strategy_config: dict, metrics_df: pd.DataFrame
):
    """Saves strategy metrics and configuration to a Parquet file."""
    if metrics_df is None:
        # logging.info(f"No metrics to save for directory: {strategy_dir}")
        return  # Exit the function if there are no metrics to save

    metrics_file = f"{strategy_dir}/metrics.parquet"

    # Extract signal and filter parameters into separate columns
    strategy_config_flat = strategy_config.copy()

    signal_params = strategy_config_flat.pop("signal_params", {})
    filter_params = strategy_config_flat.pop("filter_params", {})

    strategy_config_flat["Signal Parameters"] = [signal_params] * len(
        metrics_df
    )  # NOTE THIS IS PROBABLY THE ISSUE RIGHT?
    strategy_config_flat["Filter Parameters"] = [filter_params] * len(metrics_df)

    # Merge strategy config with metrics
    for key, value in strategy_config_flat.items():
        metrics_df[key] = value

    # Save as Parquet
    metrics_df.to_parquet(metrics_file, index=False)


def run_backtest():
    """
    1. Build all combos for (timeframe, signal_method, (filter_method, filter_params), holding_period, tp_sl, zscore_lookback).
    2. Shuffle them.
    3. For each combo, randomly pick one param dict from signal_methods[signal_method], then merge with the combo fields.
    4. Generate signals/trades/metrics or skip if already exist.
    5. Save to strategy_mapping.
    """
    # Initialize
    signal_cache = SignalCache()
    trade_cache = TradeCache()
    strategy_id = len(strategy_mapping) + 1

    try:
        # 1) Build filter combinations
        filter_combinations = [
            (method, params)
            for method, param_list in Config().filter_methods.items()
            for params in param_list
        ]

        # 2) Build all combos from your enumerations
        total_combinations = list(
            itertools.product(
                timeframes,
                signal_methods.keys(),  # e.g. "prophet_forecast", "exponential_smoothing_forecast", ...
                filter_combinations,
                holding_periods,
                tp_sl_values,
                zscore_lookback_values,
            )
        )

        # 3) Shuffle combos
        random.shuffle(total_combinations)
        print(f"Total combos after shuffle: {len(total_combinations)}")

        # 4) Loop combos
        for combo in tqdm(
            total_combinations, desc="Backtesting Progress", unit="strategy"
        ):
            (
                timeframe,
                method_name,
                (filter_method, filter_params),
                combo_holding_period,
                combo_tp_sl,
                combo_zscore,
            ) = combo

            # 4A) Grab the entire list of param dicts for this method:
            # e.g. signal_methods["prophet_forecast"] -> list of many dicts
            possible_params_list = signal_methods[method_name]

            # 4B) Randomly select exactly 1 param dict from that list
            # so each iteration picks a random config for, say, prophet or exponential_smoothing
            if not possible_params_list:
                print(f"No param list for method={method_name}, skipping.")
                continue

            sparams = random.choice(possible_params_list)

            # Force exactly ONE of {apply_log, apply_boxcox, apply_detrend} to True, others False
            transformations = ["apply_log", "apply_boxcox", "apply_detrend"]
            chosen_one = random.choice(transformations)
            for trans in transformations:
                sparams[trans] = trans == chosen_one

            # 4C) Merge the combo fields that must override the dict
            # e.g. you want the combo's holding_period, tp_sl, zscore_lookback to override sparams
            # or you can do the reverse if you want sparams to define them.
            # Here, we assume combo overrides (meaning we trust the combo's holding_period, etc.)
            new_sparams = sparams.copy()
            new_sparams["holding_period"] = combo_holding_period
            new_sparams["tp_sl"] = combo_tp_sl
            new_sparams["zscore_lookback"] = combo_zscore

            # 4D) Build the final config for storing
            strategy_label = f"strategy_{strategy_id}"
            strategy_config = {
                "timeframe": timeframe,
                "signal_method": method_name,
                "signal_params": new_sparams,  # the merged dictionary
                "filter_method": filter_method,
                "filter_params": filter_params,
                "holding_period": combo_holding_period,
                "tp_sl": combo_tp_sl,
                "zscore_lookback": combo_zscore,
            }

            # 4E) Construct file paths
            strategy_dir = os.path.join(Config().results_path, strategy_label)
            os.makedirs(strategy_dir, exist_ok=True)
            trades_file = os.path.join(strategy_dir, "trades.parquet")
            metrics_file = os.path.join(strategy_dir, "metrics.parquet")

            # 5) Skip or generate logic
            if os.path.exists(trades_file) and os.path.exists(metrics_file):
                print(f"Skipping {strategy_label}, trades+metrics exist.")
                strategy_id += 1
                continue
            elif os.path.exists(trades_file) and not os.path.exists(metrics_file):
                print(f"Found trades for {strategy_label}, computing metrics only.")
                metrics = calculate_strategy_metrics(trades_file)
                save_strategy_metrics(strategy_dir, strategy_config, metrics)
                print(f"Saved metrics for {strategy_label}.")
            else:
                print(
                    f"No trades nor metrics for {strategy_label}. Generating signals/trades/metrics..."
                )

                # 5A) generate_signals with the merged new_sparams
                signals = generate_signals(
                    timeframe=timeframe,
                    signal_method=method_name,
                    signal_params=new_sparams,
                    filter_method=filter_method,
                    filter_params=filter_params,
                    cache=signal_cache,
                )

                # 5B) Calculate trade outcomes
                trade_results = calculate_trade_outcomes(
                    df=signals,
                    holding_period=combo_holding_period,
                    tp_sl=combo_tp_sl,
                    zscore_lookback=combo_zscore,
                    cache=trade_cache,
                    timeframe=timeframe,
                )
                # 5C) Save non-zero trades
                trade_results[trade_results["signal"] != 0].to_parquet(
                    trades_file, index=False
                )
                print(f"Saved trades for {strategy_label} -> {trades_file}")

                # 5D) metrics
                metrics = calculate_strategy_metrics(trades_file)
                save_strategy_metrics(strategy_dir, strategy_config, metrics)
                print(f"Saved metrics for {strategy_label}.")

            # 6) Update mapping
            strategy_mapping[strategy_label] = strategy_config
            save_strategy_mapping()

            strategy_id += 1

    finally:
        signal_cache.close()
        trade_cache.close()
        print("Backtesting completed.")


if __name__ == "__main__":
    ### RUN BACKTESTING
    print("Starting backtesting process...")
    # logging.info("Backtesting process initiated.")
    run_backtest()
    print("Backtesting process completed.")
    # logging.info("Backtesting process completed.")
