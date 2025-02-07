import logging
import os
import pandas as pd

from config.configuration import Config
from tqdm import tqdm
from functools import lru_cache
from itertools import product

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

import os
from itertools import product

# Configure logging
logging.basicConfig(
    filename="logs/market_data_processor.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class MarketDataProcessor:
    def __init__(self):
        logging.info("Initializing MarketDataProcessor.")
        self.config = Config()  # Store Config instance
        self.raw_data_path = self.config.raw_data_path
        self.resampled_path = self.config.resampled_path

        if not os.path.exists(self.raw_data_path):
            logging.error(f"Raw data file not found: {self.raw_data_path}")
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")

        self.data = pd.read_parquet(self.raw_data_path)
        logging.info(f"Loaded raw data with shape {self.data.shape}")

        # print("Initial Raw Data:")
        # print(self.data.head())

        required_columns = {"start", "symbol", "close", "open", "high", "low", "volume"}
        missing_cols = required_columns - set(self.data.columns)
        if missing_cols:
            logging.error(f"Missing columns in raw data: {missing_cols}")
            raise KeyError(f"Missing columns: {missing_cols}")

        if not pd.api.types.is_datetime64_any_dtype(self.data["start"]):
            self.data["start"] = pd.to_datetime(self.data["start"])
            logging.info("Converted 'start' column to datetime.")

        self.data.set_index("start", inplace=True)
        logging.info("Set 'start' column as index.")

        # # Check for missing values in raw data
        # print("Missing Values Before Processing:")
        # print(self.data.isna().sum())

    def compute_returns(self):
        logging.info("Computing returns for each symbol in raw data.")
        self.data["returns"] = self.data.groupby("symbol")["close"].pct_change()
        logging.info("Returns computation completed.")

    def resample_timeframe(self, timeframe):
        logging.info(f"Resampling timeframe: {timeframe}")
        print(f"\n--- Processing {timeframe} ---")

        freq_map = {
            "ONE_MINUTE": "1min",
            "FIVE_MINUTE": "5min",
            "FIFTEEN_MINUTE": "15min",
            "THIRTY_MINUTE": "30min",
            "ONE_HOUR": "1h",
            "TWO_HOUR": "2h",
            "SIX_HOUR": "6h",
            "ONE_DAY": "1D",
        }

        if timeframe not in freq_map:
            logging.error(f"Invalid timeframe: {timeframe}")
            raise ValueError(f"Invalid timeframe: {timeframe}")

        freq = freq_map[timeframe]
        resampled_path = os.path.join(
            self.resampled_path, f"resampled_{timeframe}.parquet"
        )
        os.makedirs(os.path.dirname(resampled_path), exist_ok=True)

        # Load from cache if file exists
        if os.path.exists(resampled_path):
            logging.info(f"Loading existing resampled data: {resampled_path}")
            resampled = pd.read_parquet(resampled_path)
            print(f"Loaded existing resampled data: {resampled.shape}")
            return resampled.reset_index()

        logging.info(f"Resampling data for {timeframe}. This may take a while.")

        # Perform resampling
        resampled = (
            self.data.groupby("symbol")
            .resample(freq)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
        )

        # print("Resampled Data Before Computing Returns (First 10 rows):")
        # print(resampled.head(10))
        logging.info(f"Resampled data shape: {resampled.shape}")

        if resampled.empty:
            logging.warning(f"No data available after resampling for {timeframe}.")
            print(f"Warning: Resampled data for {timeframe} is empty.")
            return None

        # **🔹 Compute Returns Before Reindexing**
        logging.info("Computing returns before merging missing timestamps.")
        resampled["returns"] = resampled.groupby("symbol")["close"].pct_change()

        # print("Resampled Data After Computing Returns (First 10 rows):")
        # print(resampled.head(10))

        # Compute expected full range of timestamps
        start_time, end_time = self.data.index.min(), self.data.index.max()
        full_time_range = pd.date_range(start=start_time, end=end_time, freq=freq)

        unique_symbols = self.data["symbol"].unique()
        full_index = pd.DataFrame(
            product(full_time_range, unique_symbols), columns=["start", "symbol"]
        )

        # Reset index of resampled for merging
        resampled.reset_index(inplace=True)

        # **🔹 LEFT JOIN to retain existing values**
        resampled = full_index.merge(resampled, on=["start", "symbol"], how="left")

        # Set index back
        resampled.set_index(["start", "symbol"], inplace=True)

        # print("Resampled Data After Merging Missing Timestamps (First 10 rows):")
        # print(resampled.head(10))
        logging.info(f"Data after merging missing timestamps shape: {resampled.shape}")

        # **🔹 Forward-Fill, Backward-Fill, and Fill NaN**
        resampled["volume"].fillna(0, inplace=True)
        resampled = (
            resampled.groupby("symbol").fillna(method="ffill").fillna(method="bfill")
        )

        print("Missing Values After Fill:")
        print(resampled.isna().sum())

        # Save to parquet
        resampled.to_parquet(resampled_path, partition_cols=["symbol"])
        logging.info(f"Saved resampled data to {resampled_path}.")

        # print("Final Resampled Data Sample:")
        # print(resampled.head(10))

        return resampled.reset_index()


@lru_cache(maxsize=None)
def load_data_for_timeframe(timeframe: str) -> pd.DataFrame:
    """
    Load & cache the DataFrame for a given timeframe.
    """
    config = Config()
    path = os.path.join(config.resampled_path, f"resampled_{timeframe}.parquet")

    if not os.path.exists(path):
        logging.error(f"No resampled data found for timeframe: {timeframe}")
        raise FileNotFoundError(f"No resampled data found for timeframe: {timeframe}")

    return pd.read_parquet(path)


if __name__ == "__main__":
    processor = MarketDataProcessor()
    processor.compute_returns()

    for tf in tqdm(processor.config.time_intervals):
        print(f"\nProcessing timeframe: {tf}")
        data = processor.resample_timeframe(tf)
        if data is not None:
            print(data.head())
        else:
            print(f"Warning: No data generated for {tf}")
