import os
from typing import Dict

from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_num(_dir: str) -> Dict[str, int]:
    """
    get the number of images per month
    """
    hash_map = {}

    for files in tqdm(os.listdir(_dir)):
        if files.endswith(".png"):
            if files.split(".")[0].split("_")[1] not in hash_map:
                hash_map[files.split(".")[0].split("_")[1]] = 1

            else:
                hash_map[files.split(".")[0].split("_")[1]] += 1

    return hash_map


def plot_num(hash_map: Dict[str, int]) -> None:
    """
    plot the hash map
    """
    plt.figure(figsize=(20, 10))
    plt.bar(hash_map.keys(), hash_map.values())
    plt.xlabel("yyyy-mm")
    plt.ylabel("number of images")
    plt.title("Number of images per month")
    plt.show()


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="dat/img",
        help="path to the image directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img_dir = args.path
    num_dict = get_num(img_dir)
    plot_num(num_dict)
