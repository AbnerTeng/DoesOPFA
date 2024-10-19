"""
Main module for the empirical part of the project
"""
from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset.dataset import StockDataset
from .train import TrainSession
from .constants import (
    TRAIN_DATASET_DIR,
    TEST_DATASET_DIR
)
from .utils.data_utils import load_config


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--add_noise", type=bool, default=False,
        help="add noise to the image"
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="train or test"
    )
    parser.add_argument(
        "--mdl", type=str, default="bin", help="bin, multi_10"
    )
    parser.add_argument(
        "--cls", type=str, default="SH",
        help="The class of Fama French 3 factors, SH, SL, SM, BL, BH, BM, ALL"
    )
    parser.add_argument(
        "--use_spec_mdl", type=bool, default=False,
        help="Use specific model for confusing FF3 class"
    )
    parser.add_argument(
        "--spec_mdl", type=str, default="SH",
        help="Specific pre-trained model for confusing FF3 class"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config file",
        default="config/AlexNet_Thesis_session.yaml"
    )
    parser.add_argument(
        "--test_env", type=bool, default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    cfg = load_config(_args.config)
    train_label_dir = f"label/train_label_{_args.mdl}_v3.csv"
    test_label_dir = f"label/test_label_{_args.mdl}_v3.csv"
    train_dataset = StockDataset(
        TRAIN_DATASET_DIR, train_label_dir, transform, _args.add_noise, _args.cls
    )
    test_dataset = StockDataset(
        TEST_DATASET_DIR, test_label_dir, transform, _args.add_noise, _args.cls
    )
    train_size, valid_size = \
        int(0.7 * len(train_dataset)), len(train_dataset) - int(0.7 * len(train_dataset))

    train_set, valid_set = random_split(train_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)
    subsample = list(range(0, 1000, 1))
    sub_test = torch.utils.data.Subset(test_dataset, subsample)
    sub_test_loader = DataLoader(sub_test, batch_size=cfg["batch_size"], shuffle=False)
    train_session = TrainSession(
        _args.mdl, _args.cls, cfg, subsample=subsample
    )
    if _args.mode == "train":
        print("Start Training...")
        train_session.train(train_loader, valid_loader)

    elif _args.mode == "test":
        print("Start Testing...")
        train_session.test(test_loader, sub_test_loader, _args)
