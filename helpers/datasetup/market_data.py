"""
This script downloads market data for cryptocurrency products, processes it, 
and saves the data in Parquet format. It supports resuming from progress, 
fetches data in parallel using multithreading, and logs the entire process.

Key Outputs:
1. Raw Data Files: Individual CSV files for each product, saved in `data/raw_cache/`.
   - Contains OHLCV (Open, High, Low, Close, Volume) candle data for the specified time range.
2. Progress File: A CSV file in `data/progress_cache/progress.csv` that tracks the last 
   successful fetch timestamp for each product. Enables resuming the process in case of interruptions.
3. Combined Dataset: A single Parquet file with all fetched data, saved to the path 
   specified in the configuration (`Config().raw_data_path`).
   - Includes columns: `start`, `low`, `high`, `open`, `close`, `volume`, `symbol`.
4. Logs: Detailed logs saved in `logs/candles_download.log`, capturing:
   - Fetching status, retries, errors, and summary of saved files.
"""

import pandas as pd
from datetime import datetime, timedelta
import os
import time
import logging
import requests
from tqdm import tqdm
from config.configuration import Config
from coinbase.rest import RESTClient
import concurrent.futures


def setup_logging():
    """Configure logging to log both to a file and the console."""
    log_file = "logs/candles_download.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def download_market_data():
    """Main function to fetch, process, and store market data in partitioned Parquet format."""
    global combined_df
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    setup_logging()

    client = RESTClient(
        api_key=Config().name, api_secret=Config().private_key, timeout=5
    )

    max_time_span = Config().max_time_span
    max_requests_per_minute = Config().max_requests_per_minute

    os.makedirs("data/raw_cache", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/progress_cache", exist_ok=True)

    progress_file = "data/progress_cache/progress.csv"
    if not os.path.exists(progress_file):
        pd.DataFrame(columns=["product_id", "last_end_unix"]).to_csv(
            progress_file, index=False
        )

    def load_progress():
        """Load the existing progress from CSV into a dict."""
        df_prog = pd.read_csv(progress_file)
        return dict(zip(df_prog["product_id"], df_prog["last_end_unix"]))

    def save_progress(product_id, last_end_unix):
        """Save or update progress for a product."""
        df_prog = pd.read_csv(progress_file)
        if product_id in df_prog["product_id"].values:
            df_prog.loc[df_prog["product_id"] == product_id, "last_end_unix"] = (
                last_end_unix
            )
        else:
            df_prog.loc[len(df_prog)] = {
                "product_id": product_id,
                "last_end_unix": last_end_unix,
            }
        df_prog.to_csv(progress_file, index=False)

    progress_data = load_progress()

    def fetch_candles_in_range(start_datetime, end_datetime, product_id, granularity):
        """Fetch candles in chunks, resuming if partially complete."""
        current_start = start_datetime

        if product_id in progress_data:
            resume_time = datetime.fromtimestamp(progress_data[product_id])
            if resume_time > current_start:
                current_start = resume_time

        candles_data = []
        max_retries = 3

        while current_start < end_datetime:
            current_end = current_start + timedelta(seconds=max_time_span)
            if current_end > end_datetime:
                current_end = end_datetime

            start_unix = int(current_start.timestamp())
            end_unix = int(current_end.timestamp())
            retries = 0

            while retries < max_retries:
                try:
                    logging.info(
                        f"Fetching candles for {product_id} from {start_unix} to {end_unix}"
                    )
                    response = client.get_candles(
                        product_id=product_id,
                        start=start_unix,
                        end=end_unix,
                        granularity=granularity,
                    )
                    if response.candles:
                        candles_data.extend(response.candles)
                        logging.info(f"Fetched candles for {product_id}")
                        save_progress(product_id, end_unix)
                        break
                    else:
                        raise ValueError(f"Empty response from API, {response=}")
                except (requests.exceptions.RequestException, ValueError) as e:
                    retries += 1
                    logging.error(
                        f"Error fetching candles for {product_id}: {e}. "
                        f"Retry {retries}/{max_retries}"
                    )
                    time.sleep(0.1)

            current_start = current_end

        return candles_data

    def candle_to_dict(candle):
        return {
            key: getattr(candle, key, None)
            for key in ["start", "low", "high", "open", "close", "volume"]
        }

    def process_and_save_candles(candles_data, product_id):
        """Process fetched data and save it as a CSV file."""
        if not candles_data:
            logging.warning(f"No data fetched for {product_id}. Skipping.")
            return None

        columns = ["start", "low", "high", "open", "close", "volume"]
        candles_data = [candle_to_dict(x) for x in candles_data]
        df = pd.DataFrame(candles_data, columns=columns)

        df["start"] = pd.to_datetime(df["start"], unit="s")
        for col in ["low", "high", "open", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["symbol"] = product_id

        raw_cache_file = f"data/raw_cache/{product_id}_candles.csv"
        df.to_csv(raw_cache_file, index=False)
        logging.info(f"Saved candle data for {product_id} to {raw_cache_file}")

        return df

    def download_candles_for_products(product_ids, start_date_range, end_date_range):
        """Fetch and process data for multiple products (multithreading)."""
        date_format = "%d/%m/%Y"
        start_datetime = datetime.strptime(start_date_range, date_format)
        end_datetime = datetime.strptime(end_date_range, date_format)

        print(f"Starting download for {len(product_ids)} products.")
        all_data = []  # List to collect DataFrames from all threads

        def worker(product_id):
            try:
                candles = fetch_candles_in_range(
                    start_datetime, end_datetime, product_id, "ONE_MINUTE"
                )
                df = process_and_save_candles(candles, product_id)
                if df is not None:
                    all_data.append(df)  # Collect the processed DataFrame
            except Exception as e:
                logging.error(f"Failed to process {product_id}: {e}")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_requests_per_minute
        ) as executor:
            list(
                tqdm(
                    executor.map(worker, product_ids),
                    total=len(product_ids),
                    desc="Processing Products",
                )
            )

        # Combine all data after multithreading completes
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        else:
            logging.warning("No data fetched for any products.")
            return pd.DataFrame()

    product_ids = [f"{x}-USDC" for x in Config().perp_tickers]

    start_date_range = Config().start_date_range
    end_date_range = Config().end_date_range

    combined_df = download_candles_for_products(
        product_ids, start_date_range, end_date_range
    )

    if not combined_df.empty:
        combined_df.to_parquet(Config().raw_data_path, index=False)
        logging.info(f"Saved all candles data to {Config().raw_data_path}.")
        print(f"Data saved to {Config().raw_data_path}. Total rows: {len(combined_df)}")
    else:
        logging.warning("No data to save to Parquet.")


if __name__ == "__main__":
    download_market_data()
