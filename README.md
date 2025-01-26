# Crypto Market Data Pipeline

This repository contains a comprehensive pipeline for fetching, processing, and analyzing cryptocurrency market data. The project includes robust data handling, signal generation, trade simulation, and performance evaluation features.

## Features

1. **Market Data Download**:

   - Fetch OHLCV data for cryptocurrency products from Coinbase API.
   - Supports progress resumption and parallel processing.
   - Saves data in partitioned Parquet and CSV formats.
2. **Data Processing**:

   - Resamples data to various time intervals (1 minute to 1 day).
   - Computes returns and other derived features.
3. **Signal Generation**:

   - Creates trading signals based on momentum or custom strategies.
   - Caches signals for efficient reuse.
4. **Trade Simulation**:

   - Simulates trade outcomes using configurable parameters for take profit and stop loss.
   - Caches results to reduce redundant calculations.
5. **Performance Evaluation**:

   - Analyze trade performance with detailed metrics and logs.

## Key Outputs

- **Raw Data**: Partitioned OHLCV data in `data/raw_cache/`.
- **Processed Data**: Resampled data stored in Parquet format.
- **Signals**: Generated and cached trading signals in SQLite.
- **Trades**: Simulated trade outcomes saved and logged.

## Requirements

- Python 3.8 or higher
- Required libraries in `requirements.txt`

## How to Use

1. Configure API keys and parameters in `configuration.py`.
2. Run `market_data.py` to download market data.
3. Process the data using `data_processor.py`.
4. Generate signals using `signals_and_filters.py`.
5. Simulate trades and evaluate performance with `trades.py`.

## Folder Structure

- **data/**: Contains raw and processed data.
- **logs/**: Logs for debugging and monitoring.
- **scripts/**: Core scripts for various stages of the pipeline.

## Contributing

Contributions are welcome! Please submit issues or pull requests for enhancements or bug fixes.

## License

This project is licensed under the MIT License.
