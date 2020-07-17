import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt, mpld3
import os
import pickle
import numpy as np

yf.pdr_override() 

def get_symbols():
    module_dir = os.path.dirname(__file__)  # get current directory
    file_path = os.path.join(module_dir, 'all_symbols.pkl')
    with open(file_path, 'rb') as file:
        symbols = pickle.load(file)
    return symbols

all_symbols = get_symbols()

def stock_today(symbol):
    df = pdr.get_data_yahoo(symbol, period="1d", interval="1m")
    df.to_csv('data.csv')
    df = pd.read_csv('data.csv')
    close = [round(x[0],2) for x in df.iloc[:, 4:5].astype('float32').values.tolist()]

    date_ori = [x[11:-6] for x in (df.iloc[:, 0]).tolist()]

    labels = [f"""
        <table style="border: 1px solid black; font-weight:bold; font-size:larger; background-color:white">
        <tr style="border: 1px solid black;">
        <th style="border: 1px solid black;">Time:</th>
        <td style="border: 1px solid black;">{x}</td>
        </tr>
        <tr style="border: 1px solid black;">
        <th style="border: 1px solid black;">Close:</th>
        <td style="border: 1px solid black;">{y}</td>
        </tr>
        </table>
    """ for x,y in zip(date_ori, close)]

    fig, ax = plt.subplots(figsize=(11,5))
    lines = plt.plot(date_ori, close, marker="*", mec='w', mfc='blue', label = 'Close', c='lightblue')
    plt.legend()
    plt.locator_params(axis='y', nbins=6)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y")
    plt.title(f"Stock: {symbol} Date: {dt_string}")
    plt.tight_layout()
    ax.grid(False)
    ax.set_facecolor("white")
    plt.fill_between(date_ori, close, min(close), color = '#0083f2')
    tooltips = mpld3.plugins.PointHTMLTooltip(lines[0], labels=labels, voffset=10, hoffset=10)
    mpld3.plugins.connect(plt.gcf(), tooltips)
    html = mpld3.fig_to_html(fig)
    return html

def get_stock(symbol, period):
    if period == "5d":
        df = pdr.get_data_yahoo(symbol, period=period, interval="5m") 
    elif period == "1mo":
        df = pdr.get_data_yahoo(symbol, period=period, interval="90m")
    else:
        df = pdr.get_data_yahoo(symbol, period=period)

    df.to_csv('data.csv')
    df = pd.read_csv('data.csv')
    close = [round(x[0],2) for x in df.iloc[:, 4:5].astype('float32').values.tolist()]
    if period in ["5d", "1mo"]:
        date_ori = [x[:-9] for x in (df.iloc[:, 0]).tolist()]
    else:
        date_ori = [x for x in (df.iloc[:, 0]).tolist()]

    if period == "1y":
        date_ori = date_ori[0::5]
        close = close[0::5]
    elif period == "5y":
        date_ori = date_ori[0::20]
        close = close[0::20]
    elif period == "max":
        date_ori = date_ori[0::60]
        close = close[0::60]

    labels = [f"""
        <table style="border: 1px solid black; font-weight:bold; font-size:larger; background-color:white">
        <tr style="border: 1px solid black;">
        <th style="border: 1px solid black;">Date:</th>
        <td style="border: 1px solid black;">{x}</td>
        </tr>
        <tr style="border: 1px solid black;">
        <th style="border: 1px solid black;">Close:</th>
        <td style="border: 1px solid black;">{y}</td>
        </tr>
        </table>
    """ for x,y in zip(date_ori, close)]

    fig, ax = plt.subplots(figsize=(11,5))
    lines = plt.plot(date_ori, close, marker="*", mec='w', mfc='blue', label = 'Close', c='lightblue')
    plt.legend()
    plt.locator_params(axis='y', nbins=6)
    plt.title(f"Stock: {symbol} Period: {period}")
    plt.tight_layout()
    ax.grid(False)
    ax.set_facecolor("white")
    plt.fill_between(date_ori, close, min(close), color = '#0083f2')
    tooltips = mpld3.plugins.PointHTMLTooltip(lines[0], labels=labels, voffset=10, hoffset=10)
    mpld3.plugins.connect(plt.gcf(), tooltips)
    html = mpld3.fig_to_html(fig)
    return html

def get_info(symbol):
    tick = yf.Ticker(symbol)
    hist =  tick.history(period="2d")
    stock = {}
    stock["symbol"] = symbol
    stock["name"] = next((s["name"] for s in all_symbols if s["symbol"] == symbol), None)
    stock["close"] = round(hist["Close"].tolist()[-1],2)
    stock["open"] = round(hist["Open"].tolist()[-1],2)
    stock["change"] = round(hist["Close"].tolist()[-1] - hist["Close"].tolist()[0],2)
    stock["pchange"] = round((stock["change"]/hist["Close"].tolist()[0])*100,2)
    if stock["change"] > 0:
        stock["color"] = "#00d600"
    else:
        stock["color"] = "red"
    stock["volume"] = hist["Volume"].tolist()[-1]
    return stock