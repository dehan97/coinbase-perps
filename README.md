# Cryptocurrency Backtesting Engine

This repository contains a comprehensive **backtesting engine** for **cryptocurrency trading strategies**.
It is designed to conduct a **grid search over strategy parameters before deployment on **Coinbase Perpetual Futures**.
The system **heavily relies on caching** using **Parquet files** to optimize data retrieval and storage.**

---

## **🚀 Features**

### **1. Market Data Handling**

- Fetches **OHLCV** data for cryptocurrency markets using Coinbase API.
- Supports **parallel processing** and **incremental downloads**.
- Data is **partitioned and stored efficiently** in **Parquet format**.

### **2. Data Processing & Resampling**

- Converts raw market data into **different timeframes** (1 min, 5 min, 1 hour, 1 day, etc.).
- Computes **returns and volatility metrics**.
- Uses **Dask & Pandas** for optimized data handling.

### **3. Signal Generation Using Time Series Forecasting**

- Implements **machine learning-based** signal generation.
- Uses **Facebook Prophet** for time series forecasting.
- Predicts future prices and generates **buy/sell signals** dynamically.
- Supports **traditional technical indicators** like:
  - Moving Averages (MA Crossover)
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - MACD
  - Stochastic Oscillator

### **4. Trade Simulation & Execution**

- **Simulates trade outcomes** using backtested strategies.
- Applies **configurable Take Profit (TP) & Stop Loss (SL)**.
- Uses **z-score-based entry/exit conditions**.
- Optimized **trade execution** with **precomputed caches**.

### **5. Performance Evaluation**

- Calculates **key trading metrics** including:
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown (MDD)
  - Profit Factor
- Supports **custom strategy evaluation**.
