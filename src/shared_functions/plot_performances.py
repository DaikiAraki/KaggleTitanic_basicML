# Copyright 2022 Daiki Araki. All Rights Reserved.

import numpy as np
from matplotlib import pyplot as plt

def plot_performances(input_dict, path_dir):
    """
    :param input_dict: dict{name: accuracie_dict{"train": value, "verify": value}}
    :param path_dir: Path. path of working directory
    """

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)

    y_train = []
    y_verify = []
    x_label = []
    for (k, v) in input_dict.items():
        y_train.append(v["train"])
        y_verify.append(v["verify"])
        x_label.append(k)

    x = np.arange(len(x_label))  # input_dictのkey１つ毎に幅１を与える
    bar_width = 0.3

    bars_train = ax.bar(x=x, height=y_train, width=bar_width, color="orange", align="center")
    bars_verify = ax.bar(x=x + bar_width, height=y_verify, width=bar_width, color="green", align="center")
    ax.set_ylim(0., 1.)
    ax.set_xticks(x + (bar_width / 2.))
    ax.set_xticklabels(x_label)
    ax.set_title("Performances of each ML Methods")
    ax.set(ylabel="accuracy")
    ax.legend([bars_train, bars_verify], ["training", "verification"])

    fig.tight_layout()
    plt.savefig(str(path_dir) + "\\performances.png", format="png", dpi=300)
    plt.close()



