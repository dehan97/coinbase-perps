import logging
import os
import pandas as pd
import numpy as np
from config.configuration import Config
from helpers.trademgmt.signals_and_filters import (
    load_data_for_timeframe,
    generate_signals,
    SignalCache,
)
from tqdm import tqdm

##############################################################################
#                           CONFIG & LOGGING SETUP                            #
##############################################################################

logging.basicConfig(
    filename=f"{Config().logs_path}/trades.log",
    level=logging.INFO,
    format=Config().logging_format,
)


##############################################################################
#                       TRADE CACHE USING PARQUET                            #
##############################################################################
class TradeCache:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or Config().trades_cache_path

    def _get_parquet_path(self, timeframe: str, trade_params: dict) -> str:
        """Generates a unique Parquet filename using abbreviations to prevent long names."""
        # ✅ Use abbreviations for shorter file names
        param_abbrev = {
            "hp": trade_params["holding_period"],
            "tsl": trade_params["tp_sl"],
            "zl": trade_params["zscore_lookback"],
        }
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(param_abbrev.items()))
        filename = f"trades_{timeframe}_{param_str}.parquet"
        return os.path.join(self.cache_dir, filename)

    def trades_exist(self, timeframe: str, trade_params: dict) -> bool:
        """Checks if trade results already exist in the cache."""
        return os.path.exists(self._get_parquet_path(timeframe, trade_params))

    def load_trades(self, timeframe: str, trade_params: dict) -> pd.DataFrame:
        """Loads trade results from the Parquet cache."""
        file_path = self._get_parquet_path(timeframe, trade_params)
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"Trade file not found: {file_path}")

    def save_trades(self, timeframe: str, trade_params: dict, df: pd.DataFrame):
        """Saves trade results to a Parquet file."""
        file_path = self._get_parquet_path(timeframe, trade_params)
        df.to_parquet(file_path, index=False, engine="pyarrow")  # ✅ Fixed long names


##############################################################################
#                      TRADING LOGIC WITH CACHING                            #
##############################################################################
def calculate_trade_outcomes(
    df: pd.DataFrame,
    holding_period: int,
    tp_sl: float,  # Single value for both TP and SL
    zscore_lookback: int,  # Lookback window for z-score calculation
    cache: TradeCache,
    timeframe: str,
) -> pd.DataFrame:
    """
    Computes trade outcomes using z-score based TP/SL exit conditions.

    Args:
        df (pd.DataFrame): Input dataframe with 'symbol', 'start', 'close'.
        holding_period (int): Number of periods to hold the trade.
        tp_sl (float): Take profit and stop loss threshold (z-score based).
        zscore_lookback (int): Lookback window for z-score calculation.
        cache (TradeCache): Cache manager to store/retrieve trade results.
        timeframe (str): The trading timeframe.

    Returns:
        pd.DataFrame: Processed trade results.
    """
    trade_params = {
        "holding_period": holding_period,
        "tp_sl": tp_sl,
        "zscore_lookback": zscore_lookback,
    }

    def compute_trades():
        df_sorted = df.sort_values(["symbol", "start"]).reset_index(drop=True)
        df_sorted["start"] = pd.to_datetime(df_sorted["start"])

        df_sorted["returns_zscore"] = (
            df_sorted["returns"]
            - df_sorted.groupby("symbol", observed=False)["returns"].transform(
                lambda x: x.rolling(zscore_lookback, min_periods=1).mean()
            )
        ) / df_sorted.groupby("symbol", observed=False)["returns"].transform(
            lambda x: x.rolling(zscore_lookback, min_periods=1).std()
        )

        df_sorted["future_close"] = df_sorted.groupby("symbol", observed=False)[
            "close"
        ].shift(-holding_period)

        # Create shifted z-score for each step in the holding period
        future_zscores = pd.concat(
            [
                df_sorted.groupby("symbol", observed=False)["returns_zscore"].shift(-i)
                for i in range(1, holding_period + 1)
            ],
            axis=1,
        )

        # Check if TP or SL was hit in any step of the holding period
        df_sorted["tp_hit"] = (future_zscores >= tp_sl).any(axis=1)
        df_sorted["sl_hit"] = (future_zscores <= -tp_sl).any(axis=1)

        # Find the first occurrence of TP or SL within the holding period
        first_tp_index = future_zscores.ge(tp_sl).idxmax(axis=1)
        first_sl_index = future_zscores.le(-tp_sl).idxmax(axis=1)

        # Assign trade exit reason based on first event
        df_sorted["tp_sl_end_long"] = np.where(
            df_sorted["tp_hit"] & ~df_sorted["sl_hit"],
            "take_profit",
            np.where(
                df_sorted["sl_hit"] & ~df_sorted["tp_hit"],
                "stop_loss",
                np.where(
                    df_sorted["tp_hit"] & df_sorted["sl_hit"],
                    np.where(
                        first_tp_index < first_sl_index,
                        "take_profit",
                        "stop_loss",
                    ),
                    "end_of_holding_period",
                ),
            ),
        )

        df_sorted["final_return_long"] = (
            df_sorted["future_close"] - df_sorted["close"]
        ) / df_sorted["close"]

        return df_sorted[["symbol", "start", "tp_sl_end_long", "final_return_long"]]

    if cache.trades_exist(timeframe, trade_params):
        # logging.info(f"Loading cached trades for {trade_params}")
        trades_df = cache.load_trades(timeframe, trade_params)
    else:
        # logging.info(f"Computing trades for {trade_params}")
        trades_df = compute_trades()
        cache.save_trades(timeframe, trade_params, trades_df)

    if "signal" in df.columns:
        return df.merge(trades_df, on=["symbol", "start"], how="left")

    return trades_df


##############################################################################
#                                FILL CACHE                                #
##############################################################################
def fill_trades_cache(
    df: pd.DataFrame,
    timeframe: str,
    holding_periods: list,
    tp_sl_values: list,
    zscore_lookback_values: list,
    cache: TradeCache,
):
    """
    Precomputes and stores trade outcomes for all combinations of parameters.

    Args:
        df (pd.DataFrame): Market data with 'symbol', 'start', 'close'.
        timeframe (str): The trading timeframe.
        holding_periods (list): List of holding periods to test.
        tp_sl_values (list): List of TP/SL thresholds (z-score based).
        zscore_lookback_values (list): List of lookback windows for z-score calculation.
        cache (TradeCache): Cache manager to store trade results.

    Returns:
        None (stores all results in Parquet cache)
    """

    total_iterations = (
        len(holding_periods) * len(tp_sl_values) * len(zscore_lookback_values)
    )

    with tqdm(total=total_iterations, desc="Filling Trades Cache") as pbar:
        for holding_period in holding_periods:
            for tp_sl in tp_sl_values:
                for zscore_lookback in zscore_lookback_values:
                    trade_params = {
                        "holding_period": holding_period,
                        "tp_sl": tp_sl,
                        "zscore_lookback": zscore_lookback,
                    }

                    if cache.trades_exist(timeframe, trade_params):
                        logging.info(f"Skipping cached trades for {trade_params}")
                        pbar.update(1)
                        continue

                    # logging.info(f"Computing trades for {trade_params}")

                    # Compute trade outcomes
                    trade_results = calculate_trade_outcomes(
                        df=df,
                        holding_period=holding_period,
                        tp_sl=tp_sl,
                        zscore_lookback=zscore_lookback,
                        cache=cache,
                        timeframe=timeframe,
                    )

                    # Store results in cache
                    cache.save_trades(timeframe, trade_params, trade_results)

                    del trade_results
                    pbar.update(1)

    logging.info("Trade cache fully populated.")


##############################################################################
#                                USAGE EXAMPLE                                #
##############################################################################

# if __name__ == "__main__":
#     # Initialize caches
#     trade_cache = TradeCache()
#     signal_cache = SignalCache()

#     # Define parameters
#     timeframe = "ONE_HOUR"
#     signal_method = "moving_average_crossover"
#     signal_params = {"short_window": 20, "long_window": 50}
#     filter_method = "custom_volume_filter"
#     filter_params = {"volume_threshold": 10000}
#     holding_period = 5
#     tp_sl = 1.5  # TP/SL threshold in z-score terms
#     zscore_lookback = 20  # Lookback window for z-score calculation

#     # Generate signals
#     df_signals = generate_signals(
#         timeframe=timeframe,
#         signal_method=signal_method,
#         signal_params=signal_params,
#         filter_method=filter_method,
#         filter_params=filter_params,
#         cache=signal_cache,
#     )

#     # Merge signals with price data
#     df_prices = df_signals.merge(
#         load_data_for_timeframe(timeframe).reset_index()[["symbol", "start"]],
#         how="left",
#         on=["symbol", "start"],
#     )

#     # Calculate trade outcomes
#     trade_results = calculate_trade_outcomes(
#         df=df_prices,
#         holding_period=holding_period,
#         tp_sl=tp_sl,
#         zscore_lookback=zscore_lookback,
#         cache=trade_cache,
#         timeframe=timeframe,
#     )

#     # Print the results
#     print(trade_results)

#     # Close caches
#     signal_cache.close()

if __name__ == "__main__":
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
