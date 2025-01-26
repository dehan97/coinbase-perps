class Config:
    # api info
    name = ""
    private_key = ""

    # pathing
    raw_data_path = "data/raw/raw.parquet"
    resampled_path = "data/resampled"
    signals_cache_path = "data/signals_cache.db"
    filters_cache_path = "data/filters_cache.db"

    # candle download info
    start_date_range = "1/1/2022"
    end_date_range = "22/1/2025"
    granularity_seconds = 60
    max_candles = 350
    max_time_span = granularity_seconds * max_candles
    max_requests_per_minute = 10

    # Time intervals list
    time_intervals = ["", ""]

    # Instrument list
    perp_tickers = ["", ""]
