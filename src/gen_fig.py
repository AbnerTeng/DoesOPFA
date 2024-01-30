"""
figure generator
"""
import os
import warnings
from multiprocessing import Pool
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
warnings.filterwarnings("ignore")


class FigGenerator:
    """
    Generate image from tabular data
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


    def plot_stick(self, subarray: np.ndarray) -> tuple:
        """
        plot the america k-stick
        """
        subarray = self.rescale_script(subarray).astype(int)
        dat = np.zeros((49, self.date * 3), dtype=np.uint8)
        open_col_indices, close_col_indices, low_high_col_indices, ma_col_indices = \
            np.arange(0, self.date * 3 - 2, 3), \
            np.arange(2, self.date * 3, 3), \
            np.arange(1, self.date * 3 - 1, 3), \
            np.arange(0, self.date * 3, 1)
        ma_array = np.repeat(subarray[:, 4], 3)
        dat[subarray[:, 0], open_col_indices] = 255
        dat[subarray[:, 3], close_col_indices] = 255
        for i, pos in enumerate(low_high_col_indices):
            dat[subarray[i, 2]: subarray[i, 1] + 1, pos] = 255
        dat[ma_array, ma_col_indices] = 255
        dat = np.flipud(dat)
        prc_img = Image.fromarray(dat)
        return prc_img, prc_img.size


    def plot_volume(self, subarray: np.ndarray) -> tuple:
        """
        plot the volume
        """
        subarray = self.rescale_script(subarray).astype(int)
        dat = np.zeros((11, self.date * 3), dtype=np.uint8)
        vol_indices = np.arange(1, self.date * 3 - 1, 3)
        for i, pos in enumerate(vol_indices):
            dat[: subarray[i, 5], pos] = 255
        dat = np.flipud(dat)
        vol_img = Image.fromarray(dat)
        return vol_img, vol_img.size


    @staticmethod
    def concat(prc, vol) -> tuple:
        """
        concat the two images
        """
        full_img = np.concatenate((prc, vol), axis=0)
        full_img = Image.fromarray(full_img)
        return full_img, full_img.size


    def generate_single(self, args) -> None:
        """
        generate the images
        """
        k, v = args
        dates = np.array(v)
        permno_mask = self.array[:, 0] == k
        for idx, date in tqdm(enumerate(dates)):
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
            prc_img, _ = self.plot_stick(subarray)
            vol_img, _ = self.plot_volume(subarray)
            full_img, _ = self.concat(prc_img, vol_img)
            if f"{k}_{date}" in os.listdir("dat/img"):
                continue
            else:
                full_img.save(f"dat/img/{k}_{date}.png")
        return f"PEMRNO: {k} finished!"


    def generate(self) -> None:
        """
        Generate the images with multiprocessing
        """
        with Pool(10) as pool:
            args_list = list(self.hash_map.items())
            results = list(tqdm(pool.imap(self.generate_single, args_list), total=len(args_list)))
        for result in tqdm(results):
            print(result)


def parse_args() -> argparse.ArgumentParser:
    """
    parsing the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=False,
        help="whether to sample the data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prc_data = pd.read_parquet('dat/filtered_prc.parquet')
    fig_generator = FigGenerator(prc_data, do_sample=args.do_sample)
    print("start generating...")
    fig_generator.generate()
    