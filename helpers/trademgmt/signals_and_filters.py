import logging
import os
import pandas as pd
from typing import Callable, Dict, Optional
import inspect

from config.configuration import Config
from helpers.signals.time_series import TimeSeriesForecastingMixin
from helpers.datasetup.data_processor import load_data_for_timeframe

##############################################################################
#                           CONFIG & LOGGING SETUP                            #
##############################################################################

# Configure logging to write logs to a file, with a specific level & format
logging.basicConfig(
    filename=f"{Config().logs_path}/signals.log",
    level=logging.INFO,
    format=Config().logging_format,
)


##############################################################################
#                         PARQUET-BASED CACHING SETUP                         #
##############################################################################
class SignalCache:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or Config().signals_cache_path

    def _abbreviate_params(self, params: dict) -> str:
        # Dynamically abbreviate each key to its first three letters
        return "_".join(f"{k[:3]}-{params[k]}" for k in sorted(params))

    def _abbreviate_name(self, name: str) -> str:
        # Abbreviate a name by taking the first three letters of each part split by '_'
        return "_".join(part[:3] for part in name.split("_"))

    def _get_parquet_path(
        self, timeframe: str, method_name: str, signal_params: dict
    ) -> str:
        abbrev_method = self._abbreviate_name(method_name)
        abbrev_params = self._abbreviate_params(signal_params)
        filename = f"{abbrev_method}_{timeframe}_{abbrev_params}.parquet"
        return os.path.join(self.cache_dir, filename)

    def signals_exist(
        self, timeframe: str, method_name: str, signal_params: dict
    ) -> bool:
        """Checks if the Parquet file for the given signal parameters exists."""
        return os.path.exists(
            self._get_parquet_path(timeframe, method_name, signal_params)
        )

    def load_signals(
        self, timeframe: str, method_name: str, signal_params: dict
    ) -> pd.DataFrame:
        """Loads signals from the Parquet cache."""
        file_path = self._get_parquet_path(timeframe, method_name, signal_params)
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"Signal file not found: {file_path}")

    def save_signals(
        self, timeframe: str, method_name: str, signal_params: dict, df: pd.DataFrame
    ):
        """Saves signals to a partitioned Parquet file, excluding signal==0."""
        file_path = self._get_parquet_path(timeframe, method_name, signal_params)
        df_filtered = df[df["signal"] != 0]  # Remove rows where signal is 0
        df_filtered.to_parquet(file_path, index=False)

    def get_or_create_signals(
        self,
        timeframe: str,
        method_name: str,
        signal_params: dict,
        signal_func: Callable[[str, dict], pd.DataFrame],
    ) -> pd.DataFrame:
        """Retrieves or generates signals, storing them in a Parquet cache."""
        if self.signals_exist(timeframe, method_name, signal_params):
            # logging.info(
            # f"Loading cached signals for method '{method_name}' with params {signal_params} and timeframe '{timeframe}'."
            # )
            return self.load_signals(timeframe, method_name, signal_params)
        else:
            # logging.info(
            #     f"Computing signals for method '{method_name}' with params {signal_params} and timeframe '{timeframe}'."
            # )
            df = signal_func(timeframe, signal_params)
            self.save_signals(timeframe, method_name, signal_params, df)
            return df

    def get_or_create_filtered_signals(
        self,
        timeframe: str,
        method_name: str,
        signal_params: dict,
        filter_name: str,
        filter_params: dict,
        filter_func: Callable[[pd.DataFrame, dict], pd.DataFrame],
    ) -> pd.DataFrame:
        """Retrieves or computes filtered signals, then caches using Parquet."""
        combined_params = {**signal_params, **filter_params}
        # Abbreviate both the strategy and filter names
        abbrev_method = self._abbreviate_name(method_name)
        abbrev_filter = self._abbreviate_name(filter_name)
        abbrev_params = self._abbreviate_params(combined_params)
        file_path = os.path.join(
            self.cache_dir,
            f"{abbrev_method}_{abbrev_filter}_{timeframe}_{abbrev_params}.parquet",
        )

        if os.path.exists(file_path):
            # logging.info(
            #     f"Loading cached filtered signals for {method_name} with {filter_name} ({filter_params})"
            # )
            return pd.read_parquet(file_path)

        # logging.info(
        #     f"Computing filtered signals for {method_name} with filter {filter_name} ({filter_params})"
        # )
        df_signals = self.get_or_create_signals(
            timeframe, method_name, signal_params, lambda tf, sp: None
        )

        df_filtered = filter_func(df_signals, filter_params)
        df_filtered = df_filtered[df_filtered["signal"] != 0]
        df_filtered.to_parquet(file_path, index=False)

        return df_filtered

    def close(self):
        """Placeholder for closing connections (not needed for Parquet-based storage)."""
        pass


##############################################################################
#                          2) SIGNAL CREATOR CLASS                            #
##############################################################################


class SignalCreator(TimeSeriesForecastingMixin):
    """
    Contains static methods for creating signals. Inherits methods from mixin classes.
    This class is solely responsible for generating signals and does not handle caching.
    """

    pass


##############################################################################
#                          3) SIGNAL MANAGER CLASS                            #
##############################################################################


class SignalManager:
    def __init__(self, cache: SignalCache):
        self.cache = cache
        self.method_mapping = self._get_signal_methods()  # Dynamically fetch methods

    def _get_signal_methods(self):
        """
        Dynamically retrieves all signal methods from SignalCreator.
        """
        return {
            name: method
            for name, method in inspect.getmembers(
                SignalCreator, predicate=inspect.isfunction
            )
            if name.startswith("_") is False  # Avoid private methods
        }

    def get_signal(
        self, timeframe: str, method_name: str, signal_params: dict
    ) -> pd.DataFrame:
        """
        Retrieves the signal using the specified method and parameters.
        Utilizes caching to avoid recomputation.
        """
        if method_name not in self.method_mapping:
            raise ValueError(f"Unknown signal method: {method_name}")

        signal_func = self.method_mapping[method_name]

        df = self.cache.get_or_create_signals(
            timeframe, method_name, signal_params, signal_func
        )

        return df

    def close(self):
        self.cache.close()


##############################################################################
#                           4) FILTER CREATOR CLASS                           #
##############################################################################


class FilterCreator:
    """
    Manages filters that refine trade signals based on market conditions.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.filter_mapping = self._get_filter_methods()  # ✅ Dynamically load methods

    def _get_filter_methods(self):
        """
        Dynamically retrieves all filter methods from FilterCreator, excluding private methods.
        """
        return {
            name: method
            for name, method in inspect.getmembers(
                self.__class__, predicate=inspect.isfunction
            )
            if not name.startswith("_")  # ✅ Exclude private/internal methods
        }

    def _apply_filter(
        self, filter_name: str, filter_params: dict, timeframe: str
    ) -> pd.DataFrame:
        """
        Apply a dynamically registered filter method. (Private method to avoid detection)
        """
        if filter_name not in self.filter_mapping:
            raise ValueError(f"Unknown filter method: {filter_name}")

        filter_func = self.filter_mapping[filter_name]
        return filter_func(self.df, filter_params, timeframe)

    # @staticmethod
    # def custom_volume_filter(
    #     df: pd.DataFrame, filter_params: dict, timeframe: str
    # ) -> pd.DataFrame:
    #     """
    #     Removes signals where volume is below a specified threshold.
    #     """
    #     df["start"] = pd.to_datetime(df["start"])
    #     pv_data = load_data_for_timeframe(timeframe).reset_index()
    #     pv_data["start"] = pd.to_datetime(pv_data["start"])

    #     print(f"Merging filtered data with price-volume data for timeframe {timeframe}")
    #     df = pd.merge(
    #         df,
    #         pv_data,
    #         how="left",
    #         on=["symbol", "start"],
    #     )

    #     threshold = filter_params.get("volume_threshold", 10000)
    #     condition = (df["signal"] == 1) & (df["volume"] < threshold)
    #     df.loc[condition, "signal"] = 0

    #     print(f"Applied custom volume filter: {df['signal'].value_counts()}")
    #     return df

    @staticmethod
    def rolling_rsi_filter(
        df: pd.DataFrame, filter_params: dict, timeframe: str
    ) -> pd.DataFrame:
        """
        Filters signals based on rolling RSI, categorized into quantile-based buckets.
        """
        period = filter_params.get("rsi_period", 14)
        bucket_selection = filter_params.get(
            "rsi_buckets", [1, 2, 3, 4, 5]
        )  # Default: All buckets

        # Load price-volume data for the given timeframe
        pv_data = load_data_for_timeframe(timeframe).reset_index()
        pv_data["start"] = pd.to_datetime(pv_data["start"])

        print(f"Merging filtered data with price-volume data for timeframe {timeframe}")
        df = pd.merge(df, pv_data, how="left", on=["symbol", "start"])

        print(f"{df.columns=}")

        # Compute RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-9)  # Avoid division by zero
        df["rsi"] = 100 - (100 / (1 + rs))

        # Compute the number of unique RSI values
        num_unique = df["rsi"].nunique()
        print(f"{num_unique=}, {df['rsi']=}")

        # Ensure we have at least 2 bins, but at most 5 bins
        num_bins = min(5, max(2, num_unique))

        # Ensure number of labels matches num_bins
        labels = list(range(1, num_bins + 1))  # Use num_bins labels

        # Adjust qcut parameters dynamically
        try:
            df["rsi_bucket"] = pd.qcut(
                df["rsi"], num_bins, labels=labels, duplicates="drop"
            )
        except ValueError as e:
            print(f"qcut failed: {e}. Defaulting to single-bin category.")
            df["rsi_bucket"] = 3  # Assign a default category in case of failure

        # Convert to categorical before handling NaNs
        df["rsi_bucket"] = df["rsi_bucket"].astype("category")

        # Only add category 3 if it isn't already present
        if 3 not in df["rsi_bucket"].cat.categories:
            df["rsi_bucket"] = df["rsi_bucket"].cat.add_categories([3])

        # Fill NaNs before converting to int
        df["rsi_bucket"] = df["rsi_bucket"].fillna(3).astype(int)

        # Filter signals based on selected buckets
        df.loc[~df["rsi_bucket"].isin(bucket_selection), "signal"] = 0

        print(
            f"Applied rolling RSI filter with buckets {bucket_selection}: {df['signal'].value_counts()}"
        )
        return df


##############################################################################
#                             FINAL SIGNAL GENERATION                        #
##############################################################################
def generate_signals(
    timeframe: str,
    signal_method: str,
    signal_params: dict,
    filter_method: Optional[str] = None,
    filter_params: Optional[dict] = None,
    cache: Optional[SignalCache] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates signals with optional filtering, utilizing caching with Parquet storage.
    """
    print("Started generate_signals")

    if cache is None:
        print("Creating new SignalCache instance")
        cache = SignalCache(cache_dir)

    print("Initializing SignalManager")
    manager = SignalManager(cache)
    print(f"Fetching signals using {signal_method} with params {signal_params}")
    df_signals = manager.get_signal(timeframe, signal_method, signal_params)

    print("Converting 'start' column to datetime")
    df_signals["start"] = pd.to_datetime(df_signals["start"])

    print(f"Loading price-volume data for timeframe: {timeframe}")
    pv_data = load_data_for_timeframe(timeframe).reset_index()
    pv_data["start"] = pd.to_datetime(pv_data["start"])

    print("Merging signals with price-volume data")
    df_signals = pd.merge(
        df_signals,
        pv_data,
        how="left",
        on=["symbol", "start"],
    )

    if filter_method:
        print(f"Applying filter: {filter_method} with params: {filter_params}")
        # filter_creator = FilterCreator(df_signals)

        print("Fetching or creating filtered signals from cache")
        df_filtered = cache.get_or_create_filtered_signals(
            timeframe,
            signal_method,
            signal_params,
            filter_method,
            filter_params,  # ✅ Now properly iterating through different filter param values
            filter_func=lambda df, fp: FilterCreator(df)._apply_filter(
                filter_method, fp, timeframe
            ),
        )

        print("Returning filtered signals")
        return df_filtered

    else:
        print("No filter applied. Returning raw signals")
        return df_signals


##############################################################################
#                                USAGE EXAMPLE                                #
##############################################################################

# test that all signal methods are working
if __name__ == "__main__":

    def test_signal_methods():
        """
        Dynamically tests all signal methods in SignalCreator.
        Ensures each method runs without error and returns the expected DataFrame.
        """
        timeframe = "ONE_HOUR"
        test_params = {
            "moving_average_crossover": {"short_window": 5, "long_window": 10},
            "rsi_strategy": {"rsi_period": 14, "overbought": 70, "oversold": 30},
            "bollinger_bands_strategy": {"bb_period": 20, "std_dev": 2},
            "macd_strategy": {
                "short_period": 12,
                "long_period": 26,
                "signal_period": 9,
            },
            "stochastic_oscillator_strategy": {
                "k_period": 14,
                "d_period": 3,
                "overbought": 80,
                "oversold": 20,
            },
        }

        signal_methods = {
            name: method
            for name, method in inspect.getmembers(
                SignalCreator, predicate=inspect.isfunction
            )
        }

        failed_methods = []

        for method_name, method in signal_methods.items():
            if method_name.startswith("_"):
                continue  # Skip private methods

            params = test_params.get(method_name, {})

            try:
                df = method(timeframe, params)

                if not isinstance(df, pd.DataFrame):
                    raise ValueError("Method did not return a DataFrame")

                required_columns = {"symbol", "start", "signal"}
                if not required_columns.issubset(df.columns):
                    raise ValueError(
                        f"Missing required columns: {required_columns - set(df.columns)}"
                    )

                print(f"✅ {method_name} passed the test.")

            except Exception as e:
                print(f"❌ {method_name} failed: {str(e)}")
                failed_methods.append(method_name)

        if failed_methods:
            print(f"\nThe following methods failed: {', '.join(failed_methods)}")
        else:
            print("\nAll signal methods passed successfully!")

    # Run the test function
    test_signal_methods()
