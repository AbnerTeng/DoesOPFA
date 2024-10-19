"""Count the number of firms in each portfolio

Returns:
    _type_: dataframe with columns count
"""
from typing import List, Dict

from argparse import ArgumentParser, Namespace
import pandas as pd
from rich.progress import track

from ..constants import CLASSES
from ..portfolio import Portfolio


class PortfolioForFirmCount(Portfolio):
    """Class to count the number of firms in the portfolio

    Father class:
        Portfolio (class): Original Portfolio class

    Args:
        dat_path (str): path to the data
        label_type (str): type of label
    """
    def __init__(self, dat_path: str, label_type: str) -> None:
        super().__init__(dat_path, label_type)
        self.firm_count = {}


    def get_count(self, label: str, decile: int) -> Dict[str, List[int]]:
        """Get the number of firms in each portfolio

        Args:
            lb (str): label
            decile (int): decile

        Returns:
            List[int]: counter
        """
        if self.data.isna().sum().sum() == 0:
            sub_data = self.data
        else:
            sub_data = self.data[self.data[label].notna()]

        sub_data = sub_data[
            [
                'PERMNO', 'date', 'SMB_HML', 'RET', 'mktcap', label
            ]
        ]

        for ym in sub_data.date.unique():
            sub_data_ym = sub_data[sub_data.date == ym]
            sub_data_ym['label'] = sub_data_ym[label].copy()
            self.firm_count[ym] = sub_data_ym[sub_data_ym['label'] == decile].shape[0]

        return self.firm_count


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--dat_path", type=str, default="dat/temp_mc.parquet")
    parser.add_argument("--decile", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    port = PortfolioForFirmCount(args.dat_path, None)
    class_group = port.data.groupby("SMB_HML")
    full_df = pd.DataFrame()
    labels = [label for label in port.data.columns if 'P_' in label]

    for method_label in track(labels):
        for _class in CLASSES:
            port.data = class_group.get_group(_class)
            count = port.get_count(method_label, args.decile)
            count_df = pd.DataFrame(count.items(), columns=['date', 'count'])
            count_df.sort_values(by='date', inplace=True)
            full_df[f"{method_label}_{_class}"] = count_df['count']

    full_df.index = count_df['date']
    full_df.to_csv(f"dat/firm_count_label_{str(args.decile)}.csv", index=True)
