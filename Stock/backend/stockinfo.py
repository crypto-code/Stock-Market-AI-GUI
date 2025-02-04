import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpld3
import os
import pickle
import numpy as np


def get_symbols():
    """Get list of stock symbols from pickle file."""
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, "all_symbols.pkl")
    with open(file_path, "rb") as file:
        symbols = pickle.load(file)
    return symbols


all_symbols = get_symbols()


def get_stock_data(symbol, period="1d", interval=None):
    """Fetch stock data from Yahoo Finance."""
    try:
        if period == "5d":
            interval = "5m"
        elif period == "1mo":
            interval = "90m"
        df = yf.download(symbol, period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None


def process_stock_data(df, period):
    """Process stock data for plotting."""
    if df is None or df.empty:
        return None, None

    close = df["Close"].round(2).tolist()
    date_ori = pd.to_datetime(df.index).strftime("%Y-%m-%d %H:%M")

    # Resample data for different periods
    if period == "1y":
        df_resampled = df.resample("5D").last()
    elif period == "5y":
        df_resampled = df.resample("20D").last()
    elif period == "max":
        df_resampled = df.resample("60D").last()
    else:
        df_resampled = df

    close_resampled = df_resampled["Close"].round(2).tolist()
    date_resampled = pd.to_datetime(df_resampled.index).strftime("%Y-%m-%d %H:%M")

    return date_resampled, close_resampled


def create_tooltip_labels(dates, values):
    """Create HTML labels for tooltips."""
    labels = []
    for date, value in zip(dates, values):
        label = f"""
        <table style="border: 1px solid black; font-weight:bold; font-size:larger; background-color:white">
        <tr style="border: 1px solid black;">
        <th style="border: 1px solid black;">Date:</th>
        <td style="border: 1px solid black;">{date}</td>
        </tr>
        <tr style="border: 1px solid black;">
        <th style="border: 1px solid black;">Close:</th>
        <td style="border: 1px solid black;">{value}</td>
        </tr>
        </table>
        """
        labels.append(label)
    return labels


def plot_stock(date_ori, close, symbol, period=None, title=None):
    """Create interactive stock plot with tooltips."""
    fig, ax = plt.subplots(figsize=(11, 5))
    lines = plt.plot(
        date_ori, close, marker="*", mec="w", mfc="blue", label="Close", c="lightblue"
    )
    plt.legend()
    plt.locator_params(axis="y", nbins=6)
    if title:
        plt.title(title)
    plt.tight_layout()
    ax.grid(False)
    ax.set_facecolor("white")
    plt.fill_between(date_ori, close, min(close), color="#0083f2")

    # Add tooltips
    labels = create_tooltip_labels(date_ori, close)
    tooltips = mpld3.plugins.PointHTMLTooltip(
        lines[0], labels=labels, voffset=10, hoffset=10
    )
    mpld3.plugins.connect(plt.gcf(), tooltips)

    html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return html


def stock_today(symbol):
    """Get today's stock data and plot."""
    df = get_stock_data(symbol, period="1d", interval="1m")
    if df is None:
        return None

    # Access the 'Close' prices for the specific ticker
    close = df["Close"][symbol].round(2).tolist()
    date_ori = pd.to_datetime(df.index).strftime("%Y-%m-%d %H:%M")

    title = f"Stock: {symbol} Date: {datetime.now().strftime('%d/%m/%Y')}"
    return plot_stock(date_ori, close, symbol, title=title)


def get_stock(symbol, period):
    """Get historical stock data and plot."""
    df = get_stock_data(symbol, period=period)
    if df is None:
        return None

    date_ori, close = process_stock_data(df, period)
    if date_ori is None:
        return None

    title = f"Stock: {symbol} Period: {period}"
    return plot_stock(date_ori, close, symbol, title=title)


def get_info(symbol):
    """Get basic stock information."""
    try:
        tick = yf.Ticker(symbol)
        hist = tick.history(period="2d")
        if hist.empty:
            return None

        stock_info = {
            "symbol": symbol,
            "name": next(
                (s["name"] for s in all_symbols if s["symbol"] == symbol), "Unknown"
            ),
            "close": round(hist["Close"].iloc[-1], 2),
            "open": round(hist["Open"].iloc[-1], 2),
            "change": round(hist["Close"].iloc[-1] - hist["Close"].iloc[0], 2),
            "pchange": round(
                (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                / hist["Close"].iloc[0]
                * 100,
                2,
            ),
            "color": (
                "#00d600" if hist["Close"].iloc[-1] > hist["Close"].iloc[0] else "red"
            ),
            "volume": hist["Volume"].iloc[-1],
        }
        return stock_info
    except Exception as e:
        print(f"Error getting info for {symbol}: {str(e)}")
        return None
