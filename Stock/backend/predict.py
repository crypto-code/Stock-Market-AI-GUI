import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import mpld3
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import yfinance as yf
import os

sns.set()
tf.random.set_seed(1234)

num_layers = 1
size_layer = 128
timestamp = 10
epoch = 300
dropout_rate = 0.8
test_size = 30
learning_rate = 0.01


def predict_stock(symbol, period, sim, future):
    simulation_size = sim
    test_size = future

    df = yf.download(symbol, period=period, interval="1d")
    df.reset_index(inplace=True)
    df.to_csv("data.csv")
    df = pd.read_csv("data.csv")
    df = df.iloc[1:].drop(columns="Price")
    minmax = MinMaxScaler()
    df_log = minmax.fit_transform(df[["Close"]])
    df_log = pd.DataFrame(df_log, columns=["Close"])

    df_train = df_log

    class Model(tf.keras.Model):
        def __init__(self, learning_rate, num_layers, size_layer, output_size):
            super().__init__()
            self.model = tf.keras.Sequential(
                [
                    layers.LSTM(size_layer, return_sequences=True, dropout=dropout_rate)
                    for _ in range(num_layers)
                ]
                + [layers.Dense(output_size)]
            )
            self.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss="mean_squared_error",
            )

        def call(self, inputs):
            return self.model(inputs)

    def calculate_accuracy(real, predict):
        real = np.array(real)
        predict = np.array(predict)

        # Ensure the lengths match by trimming the longer array
        min_len = min(len(real), len(predict))
        real, predict = real[:min_len], predict[:min_len]

        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100

    def anchor(signal, weight):
        buffer = []
        last = signal[0]
        for i in signal[1:]:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer

    def forecast():
        model = Model(learning_rate, num_layers, size_layer, 1)
        date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

        pbar = tqdm(range(1), desc="train loop")
        loss_fn = tf.keras.losses.MeanSquaredError()  # Explicit loss function

        for _ in pbar:
            total_loss = []
            for k in range(0, df_train.shape[0] - 1, timestamp):
                index = min(k + timestamp, df_train.shape[0] - 1)
                batch_x = df_train.iloc[k:index, :].values.reshape(1, -1, 1)
                batch_y = df_train.iloc[k + 1 : index + 1, :].values.reshape(1, -1, 1)

                with tf.GradientTape() as tape:
                    predictions = model(batch_x)
                    loss = loss_fn(batch_y, predictions)  # Use explicit loss function
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )

                total_loss.append(loss.numpy())

            pbar.set_postfix(loss=np.mean(total_loss))

        future_day = test_size
        output_predict = np.zeros((df_train.shape[0] + future_day, 1))
        output_predict[0] = df_train.iloc[0].values

        upper_b = (df_train.shape[0] // timestamp) * timestamp
        for k in range(0, upper_b, timestamp):
            batch_x = df_train.iloc[k : k + timestamp, :].values.reshape(1, -1, 1)
            predictions = model(batch_x)
            output_predict[k + 1 : k + timestamp + 1] = predictions[0].numpy()

        if upper_b != df_train.shape[0]:
            batch_x = df_train.iloc[upper_b:, :].values.reshape(1, -1, 1)
            predictions = model(batch_x)
            output_predict[upper_b + 1 : df_train.shape[0] + 1] = predictions[0].numpy()
            future_day -= 1
            date_ori.append(date_ori[-1] + timedelta(days=1))

        init_value = np.zeros((1, size_layer * 2))
        for i in range(future_day):
            o = output_predict[-future_day - timestamp + i : -future_day + i]
            batch_x = o.reshape(1, -1, 1)
            predictions = model(batch_x)
            output_predict[-future_day + i] = predictions.numpy().flatten()[0]
            date_ori.append(date_ori[-1] + timedelta(days=1))

        output_predict = minmax.inverse_transform(output_predict)
        deep_future = anchor(output_predict[:, 0].tolist(), 0.4)

        return deep_future

    results = []
    for i in range(simulation_size):
        print("Simulation %d:" % (i + 1))
        results.append(forecast())

    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    for i in range(test_size):
        date_ori.append(date_ori[-1] + timedelta(days=1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format="%Y-%m-%d").tolist()

    accepted_results = []
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    for r in results:
        if (np.array(r[-test_size:]) < np.min(df["Close"])).sum() == 0 and (
            np.array(r[-test_size:]) > np.max(df["Close"]) * 2
        ).sum() == 0:
            accepted_results.append(r)

    accuracies = [
        calculate_accuracy(df["Close"].values, r[:-test_size]) for r in accepted_results
    ]

    plt.figure(figsize=(11, 5))
    for no, r in enumerate(accepted_results):
        labels = [
            f"<table style='border: 1px solid black; font-weight:bold; font-size:larger; background-color:white'><tr style='border: 1px solid black;'><th style='border: 1px solid black;'>Date:</th><td style='border: 1px solid black;'>{x}</td></tr><tr style='border: 1px solid black;'><th style='border: 1px solid black;'>Close:</th><td style='border: 1px solid black;'>{round(y,2)}</td></tr></table>"
            for x, y in zip(date_ori[::5], r[::5])
        ]
        lines = plt.plot(date_ori[::5], r[::5], label=f"forecast {no + 1}", marker="*")
        tooltips = mpld3.plugins.PointHTMLTooltip(
            lines[0], labels=labels, voffset=10, hoffset=10
        )
        mpld3.plugins.connect(plt.gcf(), tooltips)

    true_trend = plt.plot(
        df.iloc[:, 0].tolist()[::5],
        df["Close"][::5],
        label="true trend",
        c="black",
        marker="*",
    )[0]
    labels_true = [
        f"<table style='border: 1px solid black; font-weight:bold; font-size:larger; background-color:white'><tr style='border: 1px solid black;'><th style='border: 1px solid black;'>Date:</th><td style='border: 1px solid black;'>{y}</td></tr><tr style='border: 1px solid black;'><th style='border: 1px solid black;'>Close:</th><td style='border: 1px solid black;'>{round(x,2)}</td></tr></table>"
        for x, y in zip(df["Close"].tolist()[::5], df.iloc[:, 0].tolist()[::5])
    ]
    tooltips_true = mpld3.plugins.PointHTMLTooltip(
        true_trend, labels=labels_true, voffset=10, hoffset=10
    )
    mpld3.plugins.connect(plt.gcf(), tooltips_true)

    plt.legend()
    plt.title(f"Stock: {symbol} Average Accuracy: {np.mean(accuracies):.4f}")
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.xticks([])
    plt.autoscale(enable=True, axis="both", tight=None)
    os.remove("data.csv")
    html = mpld3.fig_to_html(plt.gcf())
    plt.close()
    return html
