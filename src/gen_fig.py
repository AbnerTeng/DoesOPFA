"""
figure generator
"""
from typing import Tuple
import os
import warnings
from multiprocessing import Pool

from argparse import ArgumentParser, Namespace
import gdown
import numpy as np
import pandas as pd
from PIL import Image
from rich.progress import track

warnings.filterwarnings("ignore")


class FigGenerator:
    """
    Generate image from tabular data

    Args:
        data: pd.DataFrame -> the tabular data of OHLCV & 20_MA
        do_sample: bool -> whether to sample the data into small part for development
    """
    def __init__(self, data: pd.DataFrame, do_sample: bool=False) -> None:
        self.data = data
        self.hash_map = self.data.groupby('PERMNO')['date'].unique().to_dict()
        self.array = self.data[
            ['PERMNO', 'date', 'Open', 'High', 'Low', 'Close', '20_MA', 'Volume']
        ].to_numpy()
        self.do_sample = do_sample

        if self.do_sample:
            self.sub_hash_map = {k: self.hash_map[k] for k in list(self.hash_map.keys())[:2]}

        self.date = 20


    @staticmethod
    def rescale_script(subarray: np.ndarray) -> np.ndarray:
        """
        Rescale the data to position

        Args:
            subarray: np.ndarray -> the subarray of the data

        Returns:
            np.ndarray -> the rescaled subarrays
        """
        max_index = max(subarray[:, 1].max(), subarray[:, 4].max())
        min_index = min(subarray[:, 2].min(), subarray[:, 4].min())
        max_vol = subarray[:, 5].max()

        price_ratio = 47 / (max_index - min_index)
        vol_ratio = 11 / max_vol

        subarray[:, :5] = np.round(
            (subarray[:, :5] - min_index) * price_ratio
        ) + 1
        subarray[:, 5] = np.round(subarray[:, 5] * vol_ratio) + 1
        return subarray


    def plot_sticks(self, subarray: np.ndarray) -> Tuple[Image.Image, Image.Image]:
        """
        plot the america k-stick

        Args:
            subarray: np.ndarray -> the subarray of rescaled data

        Returns:
            Tuple[Image.Image, Image.Image] -> the price and volume images

        Price Images: shape (49, 60)
        Volume Images: shape (11, 60)
        """
        subarray = self.rescale_script(subarray).astype(int)
        price_dat = np.zeros((49, self.date * 3), dtype=np.uint8)
        vol_dat = np.zeros((11, self.date * 3), dtype=np.uint8)

        open_col_indices, close_col_indices, low_high_col_indices, ma_col_indices = \
            np.arange(0, self.date * 3 - 2, 3), \
            np.arange(2, self.date * 3, 3), \
            np.arange(1, self.date * 3 - 1, 3), \
            np.arange(0, self.date * 3, 1)

        vol_indices = np.arange(1, self.date * 3 - 1, 3)
        ma_array = np.repeat(subarray[:, 4], 3)
        price_dat[subarray[:, 0], open_col_indices] = 255
        price_dat[subarray[:, 3], close_col_indices] = 255

        for i, pos in enumerate(low_high_col_indices):
            price_dat[subarray[i, 2]: subarray[i, 1] + 1, pos] = 255

        for i, pos in enumerate(vol_indices):
            vol_dat[: subarray[i, 5], pos] = 255

        price_dat[ma_array, ma_col_indices] = 255
        price_dat = np.flipud(price_dat)
        vol_dat = np.flipud(vol_dat)
        prc_img = Image.fromarray(price_dat)
        vol_img = Image.fromarray(vol_dat)

        return prc_img, vol_img


    @staticmethod
    def concat(prc: Image.Image, vol: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        concat the two images
        """
        full_img = np.concatenate((prc, vol), axis=0)
        full_img = Image.fromarray(full_img)
        return full_img, full_img.size


    def generate_single(self, args: Namespace) -> None:
        """
        generate the images
        """
        dates = np.ndarray(args.values())
        permno_mask = self.array[:, 0] == args.keys()

        for idx, date in track(enumerate(dates)):
            date_mask = self.array[:, 1] == date
            subarray = self.array[permno_mask & date_mask, 2:]

            if len(subarray) < 20:
                prev_date = dates[idx - 1] if idx > 0 else date
                prev_date_mask = self.array[:, 1] == prev_date
                prev_subarray = self.array[permno_mask & prev_date_mask, 2:]
                subarray = np.concatenate(
                    [
                        prev_subarray[-20 + len(subarray):],
                        subarray
                    ]
                )

            subarray = subarray[-20: ].astype(np.float32)
            prc_img, vol_img = self.plot_sticks(subarray)
            full_img, _ = self.concat(prc_img, vol_img)

            if f"{args.keys()}_{date}.png" not in os.listdir("fig/img"):
                full_img.save(f"fig/img/{args.keys()}_{date}.png")

        return f"PEMRNO: {args.keys()} finished!"


    def generate(self) -> None:
        """
        Generate the images with multiprocessing
        """
        with Pool(10) as pool:
            args_list = list(self.hash_map.items())
            results = list(track(pool.imap(self.generate_single, args_list), total=len(args_list)))

        for result in track(results):
            print(result)


def parsing_args() -> Namespace:
    """
    parsing the arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--do_sample",
        type=bool,
        action="store_true",
        help="whether to sample the data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    p_args = parsing_args()

    if not os.path.exists("dat/clean_data.parquet"):
        print("Downloading data...")
        gdown.download(
            "https://drive.google.com/file/d/1nt6o1cKqUid-RdVYh98KpWgO4-EV2TOX/view?usp=sharing",
            "dat/clean_data.parquet",
        )

    prc_data = pd.read_parquet('dat/clean_data.parquet')
    fig_generator = FigGenerator(prc_data, do_sample=p_args.do_sample)
    print("start generating...")
    fig_generator.generate()
