"""
Stock img dataset preparation
"""
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision

from ..utils.data_utils import filter_files, filter_imgs

class StockDataset(Dataset):
    """
    transform the data into tensor
    
    attributes:

    - path: path to the dataset
    - label_path: path to the label
    - tfm: torchvision.transforms
    - add_noise: add noise to the image
    
    Special methods:

    - torchvision.transforms
        Do no augmentation instead only transform to torch.Tensor

    - add_noise:
        Generate two random indices from (0, 59), and replace the black pixel with white pixel
    """
    def __init__(
        self,
        img_path: str,
        label_path: str,
        tfm: torchvision.transforms,
        add_noise: bool,
        _cls: str,
    ) -> None:
        self.cls = _cls
        self.label_data = filter_imgs(img_path, label_path, self.cls)
        self.imgs = filter_files(img_path, label_path, self.cls)
        self.tfm = tfm
        self.add_noise = add_noise


    def __len__(self) -> int:
        return len(self.imgs)


    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        im = Image.open(self.imgs[idx])
        im = self.tfm(im)

        if self.add_noise:
            random_idx = np.random.choice(len(im[0]), 2)

            while im[0][random_idx[0]][random_idx[1]] == 0:
                im[0][random_idx[0]][random_idx[1]] = 1

        label = self.label_data[
            self.label_data["fig_name"] == self.imgs[idx].split(".")[0].split("/")[-1]
        ]["new_label"].values[0]

        return im, label
