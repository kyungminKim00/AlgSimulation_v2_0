import re
import os
import sys
from PIL import Image
import glob
import platform
import numpy as np
import header.index_forecasting.RUNHEADER as RUNHEADER
import argparse
import datetime
import util
import pickle
from util import get_domain_on_CDSW_env
from datasets.windowing import (
    rolling_apply,
    rolling_apply_cov,
    rolling_apply_cross_cov,
    fun_mean,
    fun_cumsum,
    fun_cov,
    fun_cross_cov,
)

import matplotlib

matplotlib.use("Agg")  # Bypass the need to install Tkinter GUI framework

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_name = "./datasets/rawdata/index_data/gold_index.csv"
data = pd.read_csv(file_name)
TradeDate = data["TradeDate"].values
raw_vals = data["KS200"].values

cnt = 20
performance_samples = list()
realized_returns = list()
expected_returns = list()
forward = 1
method = "both"
mv = 1  # 5days moving average
if mv > 1:
    vals = rolling_apply(fun_mean, raw_vals, mv)
else:
    vals = raw_vals


def get_method(check_sum, expected_label, _method=None):
    if _method == "negative":
        return (check_sum == expected_label) and (expected_label == 0)
    elif _method == "positive":
        return (check_sum == expected_label) and (expected_label == 1)
    else:  # bots
        return check_sum == expected_label


for i in range(len(data["TradeDate"].values)):
    cnt = cnt + 1
    rnd_indx = np.random.randint(len(TradeDate))

    data_x = TradeDate[cnt - 20 : cnt]
    data_y = vals[cnt - 20 : cnt]

    data_x_s = TradeDate[cnt - 20 : cnt + 20]
    data_y_s = raw_vals[cnt - 20 : cnt + 20]
    mark_x_s = TradeDate[cnt]
    mark_y_s = raw_vals[cnt]
    mark_x_s_prediction = TradeDate[cnt + forward]
    mark_y_s_prediction = raw_vals[cnt + forward]

    # # Generate random data.
    # data_x = np.arange(start = 0, stop = 25, step = 1, dtype='int')
    # data_y = np.random.random(25)*6

    # Find peaks(max).
    peak_indexes = signal.argrelextrema(data_y, np.greater)
    peak_indexes = peak_indexes[0]

    # Find valleys(min).
    valley_indexes = signal.argrelextrema(data_y, np.less)
    valley_indexes = valley_indexes[0]

    # Find gradient
    try:
        expected_label = np.where(peak_indexes[-1] > valley_indexes[-1], 0, 1)

        check_sum = np.where(raw_vals[cnt] - raw_vals[cnt - 1] < 0, 0, 1)
        label = np.where(raw_vals[cnt + forward] - raw_vals[cnt] < 0, 0, 1)
        returns = ((raw_vals[cnt + forward] - raw_vals[cnt]) / raw_vals[cnt]) * 100
        realized_returns.append(returns)

        if get_method(check_sum, expected_label, _method=method):
            performance_samples.append(expected_label == label)
            cul_performance = round(
                np.sum(performance_samples) / len(performance_samples), 3
            )

            if expected_label == label:
                performance_returns = np.abs(returns)
            else:
                performance_returns = -np.abs(returns)
            expected_returns.append(performance_returns)

            # Plot main graph.
            (fig, ax) = plt.subplots()
            # ax.plot(data_x, data_y)
            ax.plot(data_x_s, data_y_s)
            plt.xticks(np.arange(0, len(data_x_s), 15))

            # Plot peaks.
            peak_x = peak_indexes
            peak_y = data_y[peak_indexes]
            ax.plot(
                peak_x,
                peak_y,
                marker="o",
                linestyle="dashed",
                color="green",
                label="Peaks",
            )

            # Plot valleys.
            valley_x = valley_indexes
            valley_y = data_y[valley_indexes]
            ax.plot(
                valley_x,
                valley_y,
                marker="o",
                linestyle="dashed",
                color="blue",
                label="Valleys",
            )

            # mark
            ax.plot(
                mark_x_s,
                mark_y_s,
                marker="o",
                linestyle="dashed",
                color="black",
                label="T",
            )
            ax.plot(
                mark_x_s_prediction,
                mark_y_s_prediction,
                marker="o",
                linestyle="dashed",
                color="red",
                label="T+{}".format(str(forward)),
            )

            # Save graph to file.
            plt.title(
                "Accuracy: {}, (BM) Return:{}, (Exptect) Return:{}".format(
                    cul_performance,
                    round(np.sum(realized_returns), 3),
                    round(np.sum(expected_returns), 3),
                )
            )
            plt.legend(loc="best")
            plt.savefig(
                "./temp/argrelextrema_MV{}_T{}_{}.png".format(mv, forward, str(cnt))
            )
    except IndexError:
        pass