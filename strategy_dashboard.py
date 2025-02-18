import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from config.configuration import Config
import webbrowser
import threading

##############################
# Configuration and Data Load
##############################

# We'll assume that 'Config' is already imported from your config.configuration
# from config.configuration import Config
results_path = Path(Config().results_path)

# Load strategy metrics
metrics_files = list(results_path.glob("strategy_*/metrics.parquet"))
df_metrics = pd.concat(
    [
        pd.read_parquet(file).assign(
            strategy_number=int(file.parent.name.split("_")[-1])
        )
        for file in metrics_files
    ],
    ignore_index=True,
)

# Load trades data
trade_files = list(results_path.glob("strategy_*/trades.parquet"))
df_trades = pd.concat(
    [
        pd.read_parquet(file).assign(
            strategy_number=int(file.parent.name.split("_")[-1])
        )
        for file in trade_files
    ],
    ignore_index=True,
)

##########################
# Max Drawdown Function
##########################


def max_dd(returns, index):
    """Assumes returns is a pandas Series with a corresponding datetime index."""
    returns = returns.astype(np.float64)  # Ensure high precision
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    returns = np.clip(returns, -0.99, 1e6)  # Ensure valid range for log1p
    log_r = np.log1p(returns)
    log_cumsum = log_r.cumsum()
    log_cumsum = np.clip(log_cumsum, -700, 50)  # Prevent extreme values
    r = np.exp(log_cumsum)

    dd = r.div(r.cummax()).sub(1)  # Negative during drawdowns
    mdd = dd.min()

    # Assign the proper index so we can identify start & end times
    dd.index = index
    r.index = index

    end = dd.idxmin()  # Trough of the worst drawdown
    start = r.loc[:end].idxmax()  # Peak before the trough

    return mdd, start, end, dd


##########################
# Dash App Setup
##########################
app = dash.Dash(__name__)
app.title = "Strategy Dashboard"

# Build lists for dropdowns
strategy_numbers = sorted(df_metrics["strategy_number"].unique())
symbols = sorted(df_trades["symbol"].unique())

##########################
# App Layout
##########################
app.layout = html.Div(
    [
        html.H1("Strategy Performance Dashboard", style={"marginBottom": 10}),
        # Top Section (Filters + Metrics)
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Strategy Number"),
                        dcc.Dropdown(
                            id="strategy-dropdown",
                            options=[
                                {"label": f"Strategy {s}", "value": s}
                                for s in strategy_numbers
                            ],
                            value=strategy_numbers[0],  # default selection
                            clearable=False,
                        ),
                    ],
                    style={"width": "200px"},
                ),
                html.Div(
                    [
                        html.Label("Select Symbol"),
                        dcc.Dropdown(
                            id="symbol-dropdown",
                            options=[{"label": sym, "value": sym} for sym in symbols],
                            value=symbols[0],  # default selection
                            clearable=False,
                        ),
                    ],
                    style={"width": "200px", "marginLeft": "30px"},
                ),
            ],
            style={"display": "flex", "marginBottom": 20},
        ),
        # Graphs
        dcc.Graph(id="price-chart"),
        dcc.Graph(id="pnl-chart"),
        dcc.Graph(id="drawdown-chart"),
        # Metrics Table
        html.H2("Overall Metrics"),
        dash_table.DataTable(
            id="overall-table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
        ),
        html.H2("Recent Metrics"),
        dash_table.DataTable(
            id="recent-table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
        ),
        html.H2("Long Metrics"),
        dash_table.DataTable(
            id="long-table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
        ),
        html.H2("Short Metrics"),
        dash_table.DataTable(
            id="short-table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
        ),
        html.H2("Other Metrics"),
        dash_table.DataTable(
            id="other-table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
        ),
    ],
    style={"margin": "20px"},
)


##########################
# Callbacks
##########################
@app.callback(
    [
        Output("price-chart", "figure"),
        Output("pnl-chart", "figure"),
        Output("drawdown-chart", "figure"),
        Output("overall-table", "data"),
        Output("overall-table", "columns"),
        Output("recent-table", "data"),
        Output("recent-table", "columns"),
        Output("long-table", "data"),
        Output("long-table", "columns"),
        Output("short-table", "data"),
        Output("short-table", "columns"),
        Output("other-table", "data"),
        Output("other-table", "columns"),
    ],
    [Input("strategy-dropdown", "value"), Input("symbol-dropdown", "value")],
)
def update_charts(strategy_number, symbol):
    """
    1. Filter df_metrics and df_trades based on strategy_number & symbol
    2. Load price data
    3. Build price chart with buy/sell signals
    4. Build PnL chart
    5. Build drawdown chart
    6. Return relevant metrics
    """

    # Filter metrics
    filtered_metrics = df_metrics[
        (df_metrics["strategy_number"] == strategy_number)
        & (df_metrics["Symbol"] == symbol)
    ]
    # Filter trades
    filtered_trades = df_trades[
        (df_trades["strategy_number"] == strategy_number)
        & (df_trades["symbol"] == symbol)
    ]

    # If we don't have price data or trades, build empty figures
    if filtered_metrics.empty or filtered_trades.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data found.")
        return empty_fig, empty_fig, empty_fig, [], []

    # Load the specific resampled file
    timeframe = filtered_metrics["timeframe"].iloc[0]
    resampled_file = f"{Config().resampled_path}/resampled_{timeframe}.parquet"
    df_price = pd.read_parquet(resampled_file)

    # Make sure the price data is multi-index => (start, symbol)
    # Then select just the symbol we want
    try:
        df_price = df_price.xs(symbol, level="symbol")
    except KeyError:
        # If we cannot find that symbol in the multi-index
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Symbol not found in price data.")
        return empty_fig, empty_fig, empty_fig, [], []

    # 1) PRICE CHART
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=df_price.index.get_level_values("start"),
            y=df_price["close"],
            mode="lines",
            name="Price",
        )
    )
    # Add buy signals
    buys = filtered_trades[filtered_trades["signal"] == 1]
    sells = filtered_trades[filtered_trades["signal"] == -1]
    fig_price.add_trace(
        go.Scatter(
            x=buys["start"],
            y=buys["close"],
            mode="markers",
            name="Buy Signals",
            marker=dict(color="green", symbol="triangle-up"),
        )
    )
    # Add sell signals
    fig_price.add_trace(
        go.Scatter(
            x=sells["start"],
            y=sells["close"],
            mode="markers",
            name="Sell Signals",
            marker=dict(color="red", symbol="triangle-down"),
        )
    )
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_price.update_layout(
        title=f"Price Movements | Strategy {strategy_number} - {symbol} | Generated: {generation_time}"
    )

    # 2) Cumulative PnL CHART
    if "cumulative_pnl" not in filtered_trades.columns:
        filtered_trades["cumulative_pnl"] = filtered_trades[
            "final_return_long"
        ].cumsum()

    fig_pnl = go.Figure()
    fig_pnl.add_trace(
        go.Scatter(
            x=filtered_trades["start"],
            y=filtered_trades["cumulative_pnl"],
            mode="lines",
            name="Cumulative PnL",
        )
    )
    fig_pnl.update_layout(
        title=f"Cumulative PnL | Strategy {strategy_number} - {symbol} | Generated: {generation_time}"
    )

    # 3) MAX DRAWDOWN CHART
    mdd, start_dd, end_dd, drawdown_series = max_dd(
        filtered_trades["final_return_long"], filtered_trades["start"]
    )

    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(
        go.Scatter(
            x=filtered_trades["start"],
            y=drawdown_series * 100,
            mode="lines",
            name="Drawdown (%)",
        )
    )
    if (start_dd in drawdown_series.index) and (end_dd in drawdown_series.index):
        fig_drawdown.add_trace(
            go.Scatter(
                x=[start_dd, end_dd],
                y=[drawdown_series[start_dd] * 100, drawdown_series[end_dd] * 100],
                mode="markers",
                name="Max Drawdown",
                marker=dict(color="red", size=10),
            )
        )
    fig_drawdown.update_layout(
        title=(
            f"Max Drawdown (%) | Strategy {strategy_number} - {symbol} | "
            f"Generated: {generation_time}<br>Max Drawdown: {mdd:.2%}"
        )
    )

    # 4) FILTERED METRICS TABLE
    # Filter the metrics to this symbol if it exists there
    show_metrics = filtered_metrics.loc[
        (filtered_metrics["Symbol"] == symbol)
        & (filtered_metrics["strategy_number"] == strategy_number)
    ].copy()

    # Convert dictionary columns to strings
    for col in ["Signal Parameters", "Filter Parameters"]:
        if col in filtered_metrics.columns:
            filtered_metrics[col] = filtered_metrics[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )

    # Round numerical columns to 3 significant figures
    for col in filtered_metrics.select_dtypes(include=[np.number]).columns:
        filtered_metrics[col] = filtered_metrics[col].apply(lambda x: round(x, 3))

    # Ensure all values are either string, number, or boolean
    for col in filtered_metrics.columns:
        filtered_metrics[col] = filtered_metrics[col].apply(
            lambda x: (
                str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x
            )
        )

    # Group columns
    groups = {
        "Overall": [col for col in filtered_metrics.columns if "Overall" in col],
        "Recent": [col for col in filtered_metrics.columns if "Recent" in col],
        "Long": [col for col in filtered_metrics.columns if "Long" in col],
        "Short": [col for col in filtered_metrics.columns if "Short" in col],
    }

    used_columns = sum(groups.values(), [])
    groups["Others"] = [
        col for col in filtered_metrics.columns if col not in used_columns
    ]

    # Ensure each table has the Symbol column for reference
    data_dict = {}
    for group_name, cols in groups.items():
        data_dict[group_name] = (
            filtered_metrics[["Symbol"] + cols].to_dict("records") if cols else []
        )
        data_dict[f"{group_name}_columns"] = (
            [{"name": col, "id": col} for col in ["Symbol"] + cols] if cols else []
        )

    # If a table group is empty, return default values
    def get_table_data(group):
        return (data_dict.get(group, []), data_dict.get(f"{group}_columns", []))

    return (
        fig_price,
        fig_pnl,
        fig_drawdown,  # Graphs
        *get_table_data("Overall"),
        *get_table_data("Recent"),
        *get_table_data("Long"),
        *get_table_data("Short"),
        *get_table_data("Others"),
    )


##########################
# Run the Dash App
##########################
def open_browser():
    """Open a web browser to the dash app."""
    webbrowser.open_new("http://127.0.0.1:8050/")


if __name__ == "__main__":
    # Start a timer that opens the browser after a short delay
    # threading.Timer(1.0, open_browser).start()

    # Now run the server (on localhost:8050 by default)
    app.run_server(debug=True, host="127.0.0.1", port=8050)
