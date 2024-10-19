"""
Generate label and map to image
"""
import os
import warnings
from typing import Literal
from itertools import chain

from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from rich.progress import track

from .utils.data_utils import shift

warnings.filterwarnings('ignore')


class LabelGenerator:
    """
    Generate label and map to image
    
    Args:
        label_type: Literal -> the type of labels (bin, multi_3, multi_10, ova)
    """
    def __init__(
        self,
        label_type: Literal["bin", "multi_3", "multi_10", "ova"]
    ) -> None:
        self.label_type = label_type


    def gen_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        generate label from dat/month_ret_org.parquet
        
        data: dat/month_ret_org.parquet
        +-------+------+--------+-------+-------+
        |PERMNO | date | TICKER | DLRET |  RET  |
        +-------+------+--------+-------+-------+
        | 10001 |199301|  GFGC  |  None | 0.000 |
        | 10001 |199302|  GFGC  |  None | 0.018 |
        | 10001 |199303|  GFGC  |  None | 0.011 |
        ...
        
        Two label types:
        - Binary Classification (bin): 1 if RET > 0 else 0
        - Multi Classification (multi_10): 0-9 based on the percentile (k=10)

        A tricky part is that we need to shift the label by 1 month

        """
        if self.label_type == "bin":
            label = np.where(data.RET > 0, 1, 0)
            data['label'] = label

        else:
            label = []
            date_group = data.groupby("date")

            for date, _ in track(date_group.groups.items()):
                subsample = date_group.get_group(date)
                interval = int(100 / int(self.label_type.split("_")[-1]))
                percentile = np.percentile(subsample.RET, np.arange(10, 100, interval))
                sub_label = np.digitize(subsample.RET, percentile)
                label.append(sub_label)

            data['label'] = list(chain(*label))

        data['new_label'] = shift(data, 'label')
        data.drop(columns="label", inplace=True)

        return data


    def get_label(self, data: pd.DataFrame, fig_path: str) -> pd.DataFrame:
        """
        get label from filtered data
        
        - fig_name example: 10001_199301
        - data: data return from gen_label
        
        output format:
        +--------------+-----------+
        |   fig_name   | new_label |
        +--------------+-----------+
        | 10001_199301 |     0     |
        | 10001_199302 |     1     |
        | 10001_199303 |     0     |
        ...
        """
        data_with_label = self.gen_label(data)
        data_with_label['fig_name'] = data_with_label['PERMNO'] + '_' + data_with_label['date']
        fig_file = [x.split(".")[0] for x in os.listdir(fig_path)]
        mask = data_with_label['fig_name'].isin(fig_file)
        filtered_data = data_with_label[mask][['fig_name', 'new_label']]

        return filtered_data


def merge_class(label: pd.DataFrame, _class: pd.DataFrame) -> pd.DataFrame:
    """
    merge label and class
    """
    merged_df = pd.merge(
        label,
        _class[['fig_name', 'SMB_HML']],
        how="inner",
        on="fig_name"
    )
    merged_df.drop_duplicates(subset=["fig_name"], inplace=True)

    return merged_df


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--dat_path", type=str, default="dat/month_ret_org.parquet")
    parser.add_argument("--dat_type", type=str, default="train")
    parser.add_argument("--label_type", type=str, default="bin")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ret_data = pd.read_parquet(args.dat_path)
    ff3_class = pd.read_parquet("dat/ff3_class.parquet")
    generator = LabelGenerator(label_type=args.label_type)
    fig_label = generator.get_label(data=ret_data, fig_path=f"dat/{args.dat_type}/")
    fig_label = merge_class(fig_label, ff3_class)
    fig_label.to_csv(f"label/{args.dat_type}_label_{args.label_type}_v2.csv")
