import logging
import os
import pandas as pd
from config.configuration import Config
from tqdm import tqdm
from functools import lru_cache

# Configure logging
logging.basicConfig(
    filename="logs/market_data_processor.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class MarketDataProcessor:
    def __init__(self):
        # Log initialization
        logging.info("Initializing MarketDataProcessor.")
        self.data = pd.read_parquet(Config().raw_data_path)
        logging.info("Loaded raw data.")

        # Convert 'start' to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.data["start"]):
            self.data["start"] = pd.to_datetime(self.data["start"])
            logging.info("Converted 'start' column to datetime.")

        # Set index to 'start'
        self.data.set_index("start", inplace=True)
        logging.info("Set 'start' column as index.")

    def compute_returns(self):
        # Compute returns on the raw data
        logging.info("Computing returns for each symbol in raw data.")
        self.data["returns"] = self.data.groupby("symbol")["close"].pct_change()
        logging.info("Returns computed for raw data.")

    def resample_timeframe(self, timeframe):
        # Resample data based on user-provided timeframe
        logging.info(f"Resampling timeframe: {timeframe}")
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
        freq = freq_map[timeframe]

        # Check if resampled file exists
        resampled_path = f"{Config().resampled_path}/resampled_{timeframe}.parquet"
        os.makedirs(os.path.dirname(resampled_path), exist_ok=True)

        # If file already exists, load it
        if os.path.exists(resampled_path):
            logging.info(f"Found existing file {resampled_path}. Loading from disk.")
            resampled = pd.read_parquet(resampled_path)
            # If 'returns' column doesn't exist, compute it now and overwrite
            if "returns" not in resampled.columns:
                logging.info("Computing returns for existing resampled data.")
                resampled["returns"] = resampled.groupby("symbol")["close"].pct_change()
                resampled.to_parquet(resampled_path, partition_cols=["symbol"])
            return resampled.reset_index()

        # Otherwise, resample and save
        logging.info(f"Resampling data for {timeframe}. This may take a while.")
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
            .dropna(subset=["open", "high", "low", "close"])
        )

        # Compute returns on the newly resampled data
        logging.info("Computing returns for newly resampled data.")
        resampled["returns"] = resampled.groupby(level=0)["close"].pct_change()

        # Save to parquet
        resampled.to_parquet(resampled_path, partition_cols=["symbol"])
        logging.info(f"Saved resampled data to {resampled_path}.")

        return resampled.reset_index()


@lru_cache(maxsize=None)
def load_data_for_timeframe(timeframe: str) -> pd.DataFrame:
    """
    Load & cache the DataFrame for a given timeframe.
    Uses a parquet file based on the timeframe string.
    """
    path = f"{Config().resampled_path}/resampled_{timeframe}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No resampled data found for timeframe: {timeframe}")
    df = pd.read_parquet(path)
    return df


if __name__ == "__main__":
    processor = MarketDataProcessor()
    processor.compute_returns()

    for tf in tqdm(Config().time_intervals):
        print(tf)
        data = processor.resample_timeframe(tf)
        print(data.head(3))
