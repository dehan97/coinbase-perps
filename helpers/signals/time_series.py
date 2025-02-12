import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from itertools import product
from helpers.datasetup.data_processor import load_data_for_timeframe
import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm
import logging


class TimeSeriesForecastingMixin:
    """
    Contains static methods for creating trading signals using time series forecasting methods.
    """

    logging.getLogger("cmdstanpy").disabled = True

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
    def _infer_frequency(index: pd.DatetimeIndex) -> str:
        inferred_freq = pd.infer_freq(index)
        return inferred_freq if inferred_freq else None

    @staticmethod
    def _apply_transformations(
        data,
        apply_log=False,
        apply_boxcox=False,
        apply_detrend=False,
    ):
        """Apply specified transformations to the data."""
        original_data = data.copy()

        if apply_log:
            data = np.log(data + 1)  # Adding 1 to avoid log(0)
        if apply_boxcox:
            data, _ = boxcox(data + 1)  # Adding 1 to ensure all data is positive
        if apply_detrend:
            result = seasonal_decompose(data, model="additive", period=1)
            data = data - result.trend
        return data, original_data

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
        apply_log = signal_params.get("apply_log", False)
        apply_boxcox = signal_params.get("apply_boxcox", False)
        apply_detrend = signal_params.get("apply_detrend", False)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df["signal"] = 0
        df["forecast"] = np.nan  # Initialize 'forecast' column

        for symbol in tqdm(df.index.get_level_values("symbol").unique()):
            symbol_data = df.xs(symbol, level="symbol").copy()
            symbol_data.index = pd.to_datetime(symbol_data.index)
            inferred_freq = TimeSeriesForecastingMixin._infer_frequency(
                symbol_data.index
            )
            if inferred_freq:
                symbol_data = symbol_data.asfreq(inferred_freq)

            # Apply transformations
            transformed_data, original_data = (
                TimeSeriesForecastingMixin._apply_transformations(
                    symbol_data["returns"],
                    apply_log,
                    apply_boxcox,
                    apply_detrend,
                )
            )
            symbol_data["returns"] = transformed_data

            for i in range(
                training_window,
                len(symbol_data) - holding_period,
                retrain_frequency,
            ):
                train_data = symbol_data["returns"].iloc[i - training_window : i]
                try:
                    model = ARIMA(train_data, order=(p, d, q))  # NOTE try SARIMA
                    fitted_model = model.fit(method="statespace")
                    forecast = fitted_model.forecast(steps=holding_period)
                    symbol_data.iloc[
                        i + holding_period - 1, symbol_data.columns.get_loc("forecast")
                    ] = forecast.values[-1]
                except Exception as e:
                    print(f"ARIMA error: {e}")
                    continue

            symbol_data["signal"] = 0
            symbol_data.loc[symbol_data["forecast"] > 0, "signal"] = 1
            symbol_data.loc[symbol_data["forecast"] < 0, "signal"] = -1

            print(f"{symbol_data['forecast'].value_counts()=}")
            print(f"{symbol_data['signal'].value_counts()=}")

            df.loc[df.index.get_level_values("symbol") == symbol, "signal"] = (
                symbol_data["signal"].values
            )

        return df.reset_index()[["symbol", "start", "signal"]]

    @staticmethod
    def sarima_forecast(timeframe: str, signal_params: dict) -> pd.DataFrame:
        """
        Generates buy/sell signals based on SARIMA forecast of returns, considering different holding periods.
        """
        p, d, q = (
            signal_params.get("p", 2),
            signal_params.get("d", 1),
            signal_params.get("q", 2),
        )
        P, D, Q, S = (
            signal_params.get("P", 1),
            signal_params.get("D", 1),
            signal_params.get("Q", 1),
            signal_params.get("S", 12),
        )  # Seasonal params
        training_window = signal_params.get("training_window", 50)
        retrain_frequency = signal_params.get("retrain_frequency", 10)
        holding_period = signal_params.get("holding_period", 1)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df["signal"] = 0
        df["forecast"] = np.nan  # Initialize 'forecast' column

        symbols = df.index.get_level_values("symbol").unique()
        for symbol in tqdm(symbols, desc="Processing Symbols", unit="symbol"):
            symbol_data = df.xs(symbol, level="symbol").copy()
            symbol_data.index = pd.to_datetime(symbol_data.index)

            inferred_freq = TimeSeriesForecastingMixin._infer_frequency(
                symbol_data.index
            )
            if inferred_freq:
                symbol_data = symbol_data.asfreq(inferred_freq)

            symbol_data["returns"], _ = (
                TimeSeriesForecastingMixin._apply_transformations(
                    symbol_data["returns"]
                )
            )

            for i in tqdm(
                range(
                    training_window,
                    len(symbol_data) - holding_period,
                    retrain_frequency,
                ),
                desc=f"Forecasting {symbol}",
                leave=False,
            ):
                train_data = symbol_data["returns"].iloc[i - training_window : i]
                try:
                    model = SARIMAX(
                        train_data,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, S),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted_model = model.fit(disp=False)
                    forecast = fitted_model.forecast(steps=holding_period)
                    symbol_data.iloc[
                        i + holding_period - 1, symbol_data.columns.get_loc("forecast")
                    ] = forecast.iloc[-1]
                except Exception as e:
                    print(f"SARIMA error for {symbol} at {i}: {e}")

            symbol_data["signal"] = np.where(symbol_data["forecast"] > 0, 1, -1)
            df.loc[df.index.get_level_values("symbol") == symbol, "signal"] = (
                symbol_data["signal"].values
            )

            print(f"{symbol_data['forecast'].value_counts()=}")
            print(f"{symbol_data['signal'].value_counts()=}")

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
        apply_log = signal_params.get("apply_log", False)
        apply_boxcox = signal_params.get("apply_boxcox", False)
        apply_detrend = signal_params.get("apply_detrend", False)

        df = load_data_for_timeframe(timeframe).copy()
        df.sort_values(["symbol", "start"], inplace=True)
        df["signal"] = 0
        df["forecast"] = np.nan  # Initialize 'forecast' column

        for symbol in tqdm(df.index.get_level_values("symbol").unique()):
            symbol_data = df.xs(symbol, level="symbol").copy()
            symbol_data.index = pd.to_datetime(symbol_data.index)
            inferred_freq = TimeSeriesForecastingMixin._infer_frequency(
                symbol_data.index
            )
            if inferred_freq:
                symbol_data = symbol_data.asfreq(inferred_freq)

            # Apply transformations
            transformed_data, original_data = (
                TimeSeriesForecastingMixin._apply_transformations(
                    symbol_data["returns"],
                    apply_log,
                    apply_boxcox,
                    apply_detrend,
                )
            )
            symbol_data["returns"] = transformed_data

            for i in range(
                training_window,
                len(symbol_data) - holding_period,
                retrain_frequency,
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
                except Exception as e:
                    print(f"ESA  error: {e}")
                    continue

            symbol_data["signal"] = 0
            symbol_data.loc[symbol_data["forecast"] > 0, "signal"] = 1
            symbol_data.loc[symbol_data["forecast"] < 0, "signal"] = -1

            print(f"{symbol_data['forecast'].value_counts()=}")
            print(f"{symbol_data['signal'].value_counts()=}")

            df.loc[df.index.get_level_values("symbol") == symbol, "signal"] = (
                symbol_data["signal"].values
            )

        return df.reset_index()[["symbol", "start", "signal"]]

    @staticmethod
    def prophet_forecast(timeframe: str, signal_params: dict) -> pd.DataFrame:
        """
        Generates buy/sell signals based on Prophet forecast of returns, considering different holding periods.
        """
        training_window = signal_params.get("training_window", 365)
        retrain_frequency = signal_params.get("retrain_frequency", 90)
        holding_period = signal_params.get("holding_period", 30)
        apply_log = signal_params.get("apply_log", False)
        apply_boxcox = signal_params.get("apply_boxcox", False)
        apply_detrend = signal_params.get("apply_detrend", False)

        df = load_data_for_timeframe(timeframe).copy().reset_index()
        df.sort_values(["symbol", "start"], inplace=True)
        df["signal"] = 0
        df["forecast"] = np.nan  # Ensure forecast is initialized as NaN

        # Set up 'ds' directly from 'start' and 'y' from 'returns'
        df["ds"] = pd.to_datetime(df["start"])
        df["y"] = df[
            "returns"
        ].copy()  # Assume 'returns' column exists and is ready to be used as 'y'

        # Apply transformations if needed
        if apply_log:
            df["y"] = np.log(df["y"] + 1)
        if apply_boxcox:
            df["y"], _ = boxcox(df["y"] + 1)
        if apply_detrend:
            result = seasonal_decompose(df["y"], model="additive", period=1)
            df["y"] = df["y"] - result.trend

        for symbol in tqdm(df["symbol"].unique(), desc="Processing symbols"):
            symbol_data = df[df["symbol"] == symbol].copy()
            symbol_data.set_index(
                "ds", inplace=True
            )  # Set 'ds' as the index for fitting

            # Infer frequency of the data
            inferred_freq = pd.infer_freq(symbol_data.index)
            if not inferred_freq:
                inferred_freq = "D"  # Default to daily frequency if not inferable

            for i in range(
                training_window, len(symbol_data) - holding_period, retrain_frequency
            ):
                train_data = symbol_data.iloc[i - training_window : i][["y"]]
                if not train_data.empty:
                    prophet_model = Prophet(
                        daily_seasonality=False, yearly_seasonality=True
                    )
                    prophet_model.fit(
                        train_data.reset_index().rename(columns={"index": "ds"})
                    )

                    future = prophet_model.make_future_dataframe(
                        periods=holding_period, freq=inferred_freq
                    )
                    forecast = prophet_model.predict(future)
                    symbol_data.loc[
                        symbol_data.index[i + holding_period - 1], "forecast"
                    ] = forecast.iloc[-1]["yhat"]

            # Assign signals based on forecast values
            symbol_data["signal"] = np.where(
                symbol_data["forecast"] > 0,
                1,
                np.where(symbol_data["forecast"] < 0, -1, 0),
            )

            # Update the main dataframe with the calculated signals and forecasts
            df.loc[df["symbol"] == symbol, "signal"] = symbol_data["signal"].values

        # Return only the required columns
        return (
            df[["symbol", "start", "signal"]].drop_duplicates().reset_index(drop=True)
        )
