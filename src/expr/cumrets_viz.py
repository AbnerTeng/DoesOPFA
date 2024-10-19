"""
Visualization for the results of portfolio
"""
from typing import List

from argparse import ArgumentParser, Namespace
import yaml
import pandas as pd
import matplotlib.pyplot as plt


def plot_eval(
    train_loss_list: List[float],
    valid_loss_list: List[float],
    train_acc_list: List[float],
    valid_acc_list: List[float],
    _type: str,
    n_epochs: int
) -> None:
    """
    plot the evaluation
    """
    _, axs = plt.subplots(2, 1, figsize=(16, 9))
    axs[0].plot(train_loss_list, label="train loss", color="red")
    axs[0].plot(valid_loss_list, label="valid loss", color="blue")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(train_acc_list, label="train acc", color="red")
    axs[1].plot(valid_acc_list, label="valid acc", color="blue")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    plt.show()
    plt.savefig(f"fig/{_type}_{n_epochs}epochs.png")


def plot_cumret(data: pd.DataFrame, cls_list: List[str]) -> None:
    """
    plot the cumulative return for each class
    """
    _, ax = plt.subplots(figsize=(14, 8))

    for item in cls_list:
        ax.plot(data['date'], data[item], label=item)

    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def plot_metric(data: pd.DataFrame, cls_list: List[str]) -> None:
    """
    plot the evaluation metrics for each class
    """
    _, ax = plt.subplots(figsize=(20, 8))

    for item in cls_list:
        ax.plot(data['ym'], data[item], label=item)

    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--cls", type=str, default="org_ff3_equal"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cumret = pd.read_csv("dat/every_data.csv", encoding="utf-8")

    with open("config/plot_config.yaml", "r", encoding="utf-8") as yml:
        plot_config = yaml.safe_load(yml)

    plot_cumret(cumret, plot_config[args.cls])
