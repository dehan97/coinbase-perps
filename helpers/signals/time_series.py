import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from itertools import product
from helpers.datasetup.data_processor import load_data_for_timeframe


class TimeSeriesForecastingMixin:
    """
    Contains static methods for creating trading signals using time series forecasting methods.
    """

    TIME_INTERVAL_MAPPING = {
        "ONE_MINUTE": "T",
        "FIVE_MINUTE": "5T",
        "FIFTEEN_MINUTE": "15T",
        "THIRTY_MINUTE": "30T",
        "ONE_HOUR": "H",
        "TWO_HOUR": "2H",
        "SIX_HOUR": "6H",
        "ONE_DAY": "D",
    }

    @staticmethod
    def infer_frequency(index: pd.DatetimeIndex) -> str:
        inferred_freq = pd.infer_freq(index)
        return inferred_freq if inferred_freq else None

    @staticmethod
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        df["returns"] = df["close"].pct_change()
        return df.dropna()

    @staticmethod
    def arima_forecast(timeframe: str, signal_params: dict) -> pd.DataFrame:
        """
        Generates buy/sell signals based on ARIMA forecast of returns, considering different holding periods.
        """
        p = signal_params.get("p", 2)
        d = signal_params.get("d", 1)
        q = signal_params.get("q", 2)
        training_window = signal_params.get("training_window", 50)
        retrain_frequency = signal_params.get("retrain_frequency", 10)
        holding_period = signal_params.get("holding_period", 1)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df = TimeSeriesForecastingMixin.compute_returns(df)
        df["signal"] = 0

        for symbol in df.index.get_level_values("symbol").unique():
            symbol_data = df.xs(symbol, level="symbol").copy()
            symbol_data.index = pd.to_datetime(symbol_data.index)
            inferred_freq = TimeSeriesForecastingMixin.infer_frequency(
                symbol_data.index
            )
            if inferred_freq:
                symbol_data = symbol_data.asfreq(inferred_freq)

            for i in range(
                training_window, len(symbol_data) - holding_period, retrain_frequency
            ):
                train_data = symbol_data["returns"].iloc[i - training_window : i]
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=holding_period)
                    symbol_data.iloc[
                        i + holding_period - 1, symbol_data.columns.get_loc("forecast")
                    ] = forecast.values[-1]
                except:
                    continue

            symbol_data["signal"] = 0
            symbol_data.loc[symbol_data["forecast"] > 0, "signal"] = 1
            symbol_data.loc[symbol_data["forecast"] < 0, "signal"] = -1

            df.loc[df.index.get_level_values("symbol") == symbol, "signal"] = (
                symbol_data["signal"].values
            )

        return df.reset_index()[["symbol", "start", "signal"]]

    @staticmethod
    def exponential_smoothing_forecast(
        timeframe: str, signal_params: dict
    ) -> pd.DataFrame:
        """
        Generates buy/sell signals based on Exponential Smoothing forecast of returns, considering different holding periods.
        """
        smoothing_level = signal_params.get("smoothing_level", 0.2)
        trend = signal_params.get("trend", None)
        seasonal = signal_params.get("seasonal", None)
        seasonal_periods = signal_params.get("seasonal_periods", None)
        training_window = signal_params.get("training_window", 50)
        retrain_frequency = signal_params.get("retrain_frequency", 10)
        holding_period = signal_params.get("holding_period", 1)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df = TimeSeriesForecastingMixin.compute_returns(df)
        df["signal"] = 0

        for symbol in df.index.get_level_values("symbol").unique():
            symbol_data = df.xs(symbol, level="symbol").copy()
            symbol_data.index = pd.to_datetime(symbol_data.index)
            inferred_freq = TimeSeriesForecastingMixin.infer_frequency(
                symbol_data.index
            )
            if inferred_freq:
                symbol_data = symbol_data.asfreq(inferred_freq)

            for i in range(
                training_window, len(symbol_data) - holding_period, retrain_frequency
            ):
                train_data = symbol_data["returns"].iloc[i - training_window : i]
                try:
                    model = ExponentialSmoothing(
                        train_data,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods,
                    )
                    fitted_model = model.fit(smoothing_level=smoothing_level)
                    forecast = fitted_model.forecast(steps=holding_period)
                    symbol_data.iloc[
                        i + holding_period - 1, symbol_data.columns.get_loc("forecast")
                    ] = forecast.values[-1]
                except:
                    continue

            symbol_data["signal"] = 0
            symbol_data.loc[symbol_data["forecast"] > 0, "signal"] = 1
            symbol_data.loc[symbol_data["forecast"] < 0, "signal"] = -1

            df.loc[df.index.get_level_values("symbol") == symbol, "signal"] = (
                symbol_data["signal"].values
            )

        return df.reset_index()[["symbol", "start", "signal"]]
