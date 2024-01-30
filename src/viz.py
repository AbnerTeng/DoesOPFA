"""
Visualization for the results of portfolio
"""
from argparse import ArgumentParser
import yaml
import pandas as pd
import matplotlib.pyplot as plt


def plot_cumret(data: pd.DataFrame, cls: list) -> None:
    """
    plot the cumulative return for each class
    """
    _, ax = plt.subplots(figsize=(14, 8))
    for column in cls:
        ax.plot(data['date'], data[column], label=column)
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def parse_args() -> ArgumentParser:
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
    