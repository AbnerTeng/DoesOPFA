"""
Generate label and map to image
"""
import os
import warnings
from itertools import chain
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils.data_utils import shift
warnings.filterwarnings('ignore')


class LabelGenerator:
    """
    Generate label and map to image
    
    attribute:
    - label_type: bin, multi, ova_h, ova_l
    """
    def __init__(self, label_type: str) -> None:
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
        
        Three label type:
        - Binary Classification (bin): 1 if RET > 0 else 0
        - Multi Classification (multi_10): 0-9 based on the percentile (k=10)
        - Multi CLassification (multi_3): 0-2 based on the percentile (k=3)
        - One vs All (ova): 1 if RET is in the highest/lowest 10% else 0
        
        A tricky part is that we need to shift the label by 1 month
                
        """
        if self.label_type == "bin":
            label = np.where(data.RET > 0, 1, 0)
            data['label'] = label
        else:
            label = []
            date_group = data.groupby("date")
            for date in tqdm(date_group.groups.keys()):
                subsample = date_group.get_group(date)
                interval = 10 if self.label_type.split("_")[-1] == "10" else 33
                percentile = np.percentile(subsample.RET, np.arange(10, 100, interval))
                sub_label = np.digitize(subsample.RET, percentile)
                if self.label_type.split("_")[0] == "multi":
                    label.append(sub_label)
                # elif self.label_type == "ova_h":
                #     ova_label = np.where(sub_label == 9, 1, 0)
                # elif self.label_type == "ova_l":
                #     ova_label = np.where(sub_label == 0, 1, 0)
                # label.append(ova_label)
            data['label'] = list(chain(*label))
        data['new_label'] = shift(data, 'label')
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
        data['fig_name'] = data['PERMNO'] + '_' + data['date']
        fig_file = os.listdir(fig_path)
        fig_file = [x.split(".")[0] for x in fig_file]
        mask = data['fig_name'].isin(fig_file)
        filtered_df = data[mask][['fig_name', 'new_label']]
        return filtered_df


def merge_class(label: pd.DataFrame, _class: pd.DataFrame) -> pd.DataFrame:
    """
    merge label and class
    """
    df = pd.merge(
        label, _class[['fig_name', 'SMB_HML']],
        how="inner", on="fig_name"
    )
    df.drop_duplicates(subset=["fig_name"], inplace=True)
    return df


def parse_args() -> ArgumentParser:
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
    ff3_class = pd.read_csv("dat/ff3_class.csv")
    generator = LabelGenerator(label_type=args.label_type)
    full_data = generator.gen_label(ret_data)
    fig_label = generator.get_label(data=full_data, fig_path=f"dat/{args.dat_type}/")
    fig_label = merge_class(fig_label, ff3_class)
    fig_label.to_csv(f"label/{args.dat_type}_label_{args.label_type}_v2.csv")
    