import pandas as pd
from helpers.datasetup.data_processor import load_data_for_timeframe


class TechnicalIndicatorsMixin:
    """
    Contains static methods for creating signals. Inherits methods from mixin classes.
    This class is solely responsible for generating signals and does not handle caching.
    """

    @staticmethod
    def moving_average_crossover(timeframe: str, signal_params: dict) -> pd.DataFrame:
        """
        Generates buy/sell signals based on moving average crossovers.
        """
        short_window = signal_params.get("short_window", 10)
        long_window = signal_params.get("long_window", 50)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df.reset_index(inplace=True)

        df["short_ma"] = df.groupby("symbol", observed=False)["close"].transform(
            lambda x: x.rolling(window=short_window, min_periods=1).mean()
        )
        df["long_ma"] = df.groupby("symbol", observed=False)["close"].transform(
            lambda x: x.rolling(window=long_window, min_periods=1).mean()
        )

        df["signal"] = 0
        df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1
        df.loc[df["short_ma"] < df["long_ma"], "signal"] = -1

        return df[["symbol", "start", "signal"]]

    @staticmethod
    def rsi_strategy(timeframe: str, signal_params: dict) -> pd.DataFrame:
        """
        Generates buy/sell signals based on the Relative Strength Index (RSI).
        A buy signal (1) occurs when RSI crosses below the oversold threshold (e.g., 30).
        A sell signal (-1) occurs when RSI crosses above the overbought threshold (e.g., 70).
        """
        period = signal_params.get("rsi_period", 14)
        overbought = signal_params.get("overbought", 70)
        oversold = signal_params.get("oversold", 30)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df.reset_index(inplace=True)

        df["delta"] = df.groupby("symbol", observed=False)["close"].diff()
        df["gain"] = df["delta"].apply(lambda x: x if x > 0 else 0)
        df["loss"] = df["delta"].apply(lambda x: -x if x < 0 else 0)

        df["avg_gain"] = df.groupby("symbol", observed=False)["gain"].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )
        df["avg_loss"] = df.groupby("symbol", observed=False)["loss"].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )

        df["rs"] = df["avg_gain"] / df["avg_loss"]
        df["rsi"] = 100 - (100 / (1 + df["rs"]))

        df["signal"] = 0
        df.loc[df["rsi"] < oversold, "signal"] = 1  # Buy
        df.loc[df["rsi"] > overbought, "signal"] = -1  # Sell

        return df[["symbol", "start", "signal"]]

    @staticmethod
    def bollinger_bands_strategy(timeframe: str, signal_params: dict) -> pd.DataFrame:
        """
        Generates buy/sell signals based on Bollinger Bands.
        A buy signal (1) occurs when the price crosses below the lower band.
        A sell signal (-1) occurs when the price crosses above the upper band.
        """
        period = signal_params.get("bb_period", 20)
        std_dev = signal_params.get("std_dev", 2)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df.reset_index(inplace=True)

        df["rolling_mean"] = df.groupby("symbol", observed=False)["close"].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )
        df["rolling_std"] = df.groupby("symbol", observed=False)["close"].transform(
            lambda x: x.rolling(window=period, min_periods=1).std()
        )

        df["upper_band"] = df["rolling_mean"] + (df["rolling_std"] * std_dev)
        df["lower_band"] = df["rolling_mean"] - (df["rolling_std"] * std_dev)

        df["signal"] = 0
        df.loc[df["close"] < df["lower_band"], "signal"] = 1  # Buy
        df.loc[df["close"] > df["upper_band"], "signal"] = -1  # Sell

        return df[["symbol", "start", "signal"]]

    @staticmethod
    def macd_strategy(timeframe: str, signal_params: dict) -> pd.DataFrame:
        """
        Generates buy/sell signals based on the Moving Average Convergence Divergence (MACD).
        A buy signal (1) occurs when the MACD line crosses above the signal line.
        A sell signal (-1) occurs when the MACD line crosses below the signal line.
        """
        short_period = signal_params.get("short_period", 12)
        long_period = signal_params.get("long_period", 26)
        signal_period = signal_params.get("signal_period", 9)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df.reset_index(inplace=True)

        df["ema_short"] = df.groupby("symbol", observed=False)["close"].transform(
            lambda x: x.ewm(span=short_period, adjust=False).mean()
        )
        df["ema_long"] = df.groupby("symbol", observed=False)["close"].transform(
            lambda x: x.ewm(span=long_period, adjust=False).mean()
        )

        df["macd"] = df["ema_short"] - df["ema_long"]
        df["macd_signal"] = df.groupby("symbol", observed=False)["macd"].transform(
            lambda x: x.ewm(span=signal_period, adjust=False).mean()
        )

        df["signal"] = 0
        df.loc[df["macd"] > df["macd_signal"], "signal"] = 1  # Buy
        df.loc[df["macd"] < df["macd_signal"], "signal"] = -1  # Sell

        return df[["symbol", "start", "signal"]]

    @staticmethod
    def stochastic_oscillator_strategy(
        timeframe: str, signal_params: dict
    ) -> pd.DataFrame:
        """
        Generates buy/sell signals based on the Stochastic Oscillator.
        A buy signal (1) occurs when %K crosses above %D below the oversold threshold.
        A sell signal (-1) occurs when %K crosses below %D above the overbought threshold.
        """
        k_period = signal_params.get("k_period", 14)
        d_period = signal_params.get("d_period", 3)
        overbought = signal_params.get("overbought", 80)
        oversold = signal_params.get("oversold", 20)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df.reset_index(inplace=True)

        df["low_min"] = df.groupby("symbol", observed=False)["low"].transform(
            lambda x: x.rolling(window=k_period, min_periods=1).min()
        )
        df["high_max"] = df.groupby("symbol", observed=False)["high"].transform(
            lambda x: x.rolling(window=k_period, min_periods=1).max()
        )

        df["%K"] = (
            100 * (df["close"] - df["low_min"]) / (df["high_max"] - df["low_min"])
        )
        df["%D"] = df.groupby("symbol", observed=False)["%K"].transform(
            lambda x: x.rolling(window=d_period, min_periods=1).mean()
        )

        df["signal"] = 0
        df.loc[(df["%K"] > df["%D"]) & (df["%K"] < oversold), "signal"] = 1  # Buy
        df.loc[(df["%K"] < df["%D"]) & (df["%K"] > overbought), "signal"] = -1  # Sell

        return df[["symbol", "start", "signal"]]
