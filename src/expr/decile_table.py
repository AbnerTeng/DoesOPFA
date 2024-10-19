"""
Get decile table

Check the average return of stocks from a specific decile
"""
import os
from typing import List, Union

from argparse import ArgumentParser, Namespace
import pandas as pd

from ..portfolio import (
    Portfolio,
    readable_result,
    get_hl_portfolio,
)
from ..utils.data_utils import calculate_sharpe


class GetDecileTable(Portfolio):
    """Get Deile Table of binary classification

    Args:
        data_path (str): path to the data
        label_type (str): type of label
        rf_data (pd.DataFrame): risk-free rate data
    """

    def __init__(
        self,
        data_path: str,
        label_type: str,
        rf_data: pd.DataFrame
    ) -> None:
        super().__init__(data_path, label_type)
        self.rf_data = rf_data

    def get_bin_decile(self, args: Namespace) -> Union[pd.Series, List[float]]:
        """
        Get decile table
        """
        profit = self.form_portfolio(
            args.weighted_method, args.gen_benchmark, args.get_full
        )
        return profit

    def get_mc_decile(self, args: Namespace) -> Union[pd.Series, List[float]]:
        """
        Get decile table
        """
        profit = self.form_portfolio(
            args.weighted_method, args.gen_benchmark, args.get_full
        )
        return profit

    def construct_table(self, args: Namespace) -> Union[pd.Series, List[float], None]:
        """
        Construct the decile table
        """
        func_mp = {
            "bin": self.get_bin_decile,
            "mc": self.get_mc_decile
        }
        profit = func_mp[args.dat_path.split(".")[0].split("_")[-1]](args)
        profit.set_index('date', inplace=True)
        readable_profit = readable_result(profit)
        readable_profit = get_hl_portfolio(readable_profit)
        returns = readable_profit.copy().drop(columns=['class'])

        if args.perf == "ret":
            return returns.mean()

        if args.perf == "sharpe":
            sharpe = []
            rf = self.rf_data['RF'].apply(lambda x: x / 100)
            rf = rf.loc[returns.index]
            sharpe.append(calculate_sharpe(returns[col], rf) for col in returns.columns)

            return sharpe

        else:
            raise ValueError("Invalid performance measure")


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--dat_path", type=str, default="dat/temp_mc.parquet")
    parser.add_argument("--label_type", type=str, default="P_KJX_SH(y=1)")
    parser.add_argument("--weighted_method", type=str, default="value_weight")
    parser.add_argument("--get_full", action="store_true")
    parser.add_argument("--gen_benchmark", action="store_true")
    parser.add_argument("--perf", type=str, default="ret")

    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    mp = {
        'equal_weight': 'eq',
        'value_weight': 'val',
    }
    table_path_map = {
        "bin": f"decile_table_{_args.perf}_{mp[_args.weighted_method]}.csv",
        "mc": f"mc_table_{_args.perf}_{mp[_args.weighted_method]}.csv",
    }
    ff5data = pd.read_csv("dat/ff5_dat.csv")
    ff5data.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    ff5data.date = pd.to_datetime(ff5data.date, format='%Y%m')
    ff5data.set_index('date', inplace=True)
    getter = GetDecileTable(_args.dat_path, _args.label_type, ff5data)

    if table_path_map[_args.dat_path.split(".")[0].split("_")[-1]] not in os.listdir('dat'):
        full_table = pd.DataFrame()
    else:
        full_table = pd.read_csv(
            f"dat/{table_path_map[_args.dat_path.split('.')[0].split('_')[-1]]}",
            index_col=0
        )

    spec_decile = getter.construct_table(_args)
    full_table[f"{_args.label_type}_ALL"] = spec_decile
    full_table.to_csv(
        f"dat/{table_path_map[_args.dat_path.split('.')[0].split('_')[-1]]}",
        index=True
    )
    print("Table saved!")
