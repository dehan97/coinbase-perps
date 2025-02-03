import os
from itertools import product


class Config:
    # api info
    name = "INSERT HERE"
    private_key = "INSERT HERE"

    # pathing
    logs_path = "logs"
    results_path = "results"
    raw_data_path = "data/raw/raw.parquet"
    resampled_path = "data/resampled"
    signals_cache_path = "signals_cache/"
    trades_cache_path = "trades_cache/"
    results_path = "results/"

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(resampled_path, exist_ok=True)
    os.makedirs(signals_cache_path, exist_ok=True)
    os.makedirs(trades_cache_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # log format
    logging_format = "%(asctime)s [%(levelname)s] %(message)s"

    # trade cost
    one_way_trade_costs = 0.006 + 0.001  # fee + slippage

    # candle download info
    start_date_range = "1/1/2022"
    end_date_range = "22/1/2025"
    granularity_seconds = 60
    max_candles = 350
    max_time_span = granularity_seconds * max_candles
    max_requests_per_minute = 10

    ### Strategy Parameters ###
    # Time intervals list
    time_intervals = [
        "ONE_DAY",
    ]

    holding_periods = [x for x in range(50, 101, 5)]
    tp_sl_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    zscore_lookback_values = [10, 50, 100, 250, 500, 1000]
    signal_methods = {
        "moving_average_crossover": [
            {"short_window": short, "long_window": long}
            for short, long in product([5, 10, 20, 30, 50], [50, 100, 200])
            if short < long  # Ensure short_window is always smaller than long_window
        ]
    }

    filter_methods = {
        "custom_volume_filter": [
            {"volume_threshold": vol}
            for vol in [10000, 20000, 50000, 100000, 250000, 500000, 750000, 1000000]
        ],
        None: [None],  # Keep an option for no filter
    }

    use_absolute_returns_options = [True, False]

    # Instrument list
    perp_tickers = ["BTC"]
