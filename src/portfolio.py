"""
Portfolio former v3
"""
import os
import warnings

from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from rich.progress import track

from .utils.weight_utils import (
    gen_decile,
    get_benchmark,
    get_full_decile,
    high_minus_low
)
from .constants import CLASSES

warnings.filterwarnings("ignore")


class Portfolio:
    """
    Form Portfolio
    """
    def __init__(self, dat_path: str, label_type: str) -> None:
        """
        * self.data
            ** mdl_output_bin.parquet
            ** mdl_output_multi_label.parquet
        
        * prob_cols -> all probability columns
        ...
            PERMNO    date     mktcap     bm   SMB HML  SMB_HML  mktcap      RET     P_KJX_SH(y=1) ..
        0  10001    199301    15120.0  0.604   S   M      SM     44.0    0.000000     [p1, p2]
        1  10001    199302    15390.0  0.691   S   M      SM     23.0    0.017857     [p1, p2]
        2  10001    199303   15318.75  0.691   S   M      SM    137.0    0.011053     [p1, p2]
        3  10001    199304   16393.75  0.691   S   M      SM    473.0    0.070175     [p1, p2]
        4  10001    199305  16259.375  0.753   S   M      SM    126.0   -0.008197     [p1, p2]
        """
        self.dat_path = dat_path

        try:
            self.data = pd.read_parquet(self.dat_path)

        except FileNotFoundError:
            print(f"{self.dat_path} not found")

        if "date" not in self.data.columns:
            self.data.rename(columns={"next_date": "date"}, inplace=True)

        if self.data.isna().any().any():
            self.data.replace("None", np.nan, inplace=True)

        self.prob_cols = [col for col in self.data.columns if col.split("_")[0] == "P"]
        self.label_type = label_type
        self.profit = {}


    def form_portfolio(
        self,
        weighted_method: str,
        gen_benchmark: bool,
        get_full: bool
    ) -> pd.DataFrame:
        """
        Form portfolio with different weighted method
        
        * equal_weight
        * value_weight
        
        ---------------
        
        Portfolio
        
        * Benchmark
            ** Buy and Hold (BnH) all stocks
        
        * H-L Portfolio
            ** Binary Classification (KJX)
                Long the top decile and short the bottom decile

            ** Multi-Class Classification (MC)
                Long the (y=10) and short the (y=1)
        
        ---------------
        
        Return

        * pd.DataFrame with profit from different classification strategies
            ** date
            ** profit
        """
        if self.data.isna().sum().sum() == 0:
            sub_data = self.data
        else:
            sub_data = self.data[self.data[self.label_type].notna()]
        sub_data = sub_data[
            [
                'PERMNO', 'date', 'SMB_HML', 'RET', 'mktcap', self.label_type
            ]
        ]

        for ym in sub_data.date.unique():
            sub_data_ym = sub_data[sub_data.date == ym]

            if self.label_type.split("_")[1] == "KJX":
                sub_data_ym['label'] = gen_decile(sub_data_ym[self.label_type], 10)

                if get_full:
                    self.profit[ym] = get_full_decile(sub_data_ym, weighted_method)

            elif self.label_type.split("_")[1] == "MC":
                sub_data_ym['label'] = sub_data_ym[self.label_type].copy()

                if get_full:
                    self.profit[ym] = get_full_decile(sub_data_ym, weighted_method)
                else:
                    self.profit[ym] = high_minus_low(sub_data_ym, weighted_method)

            else:
                raise ValueError("Please specify the label type")

            if gen_benchmark:
                self.profit[ym] = get_benchmark(sub_data_ym, weighted_method)

        profit_df = pd.DataFrame(
            self.profit.items(), columns=['date', 'profit']
        )
        profit_df['class'] = sub_data_ym['SMB_HML'].values[0]
        profit_df.sort_values(by='date', inplace=True)

        return profit_df


def readable_result(data: pd.DataFrame) -> pd.DataFrame:
    """
    change layout from
    +------+--------------+
    | date | profit_list  |
    +------+--------------+
    |200101|[p1, ..., p10]|
    ...
    of size (264, 1)
    
    to
    
    +----+----+ 
    | p1 | p2 |...
    +----+----+
    | a1 | a2 |
    | b1 | b2 |
    ...
    of size (264, 10)
    """
    output = pd.DataFrame(data.profit.tolist(), index=data.index)
    output.columns = [f"decile_{i+1}" for i in range(10)]
    output['class'] = data['class']
    return output


def get_hl_portfolio(data: pd.DataFrame) -> pd.DataFrame:
    """
    Get H-L Portfolio, ret = decile_10 - decile_1
    """
    data["HL"] = data[f"decile_{10}"] - data[f"decile_{1}"]
    return data


def merge_all_label(
    dat_path: str,
    data: pd.DataFrame,
    label: str,
    weight: str,
    gen_benchmark: bool,
    test_env: bool
) -> None:
    """
    Merge all cum_profit
    
    * series: pd.Series
        ** the label_type column of profit
    """
    name = f"{label}_{weight}_bench" if gen_benchmark else f"{label}_{weight}"

    if not os.path.exists(dat_path):
        dat = pd.DataFrame()
        dat = pd.concat([dat, data], axis=1)
        dat.rename(columns={"HL": name}, inplace=True)

    else:
        dat = pd.read_csv(dat_path, index_col=0)
        dat[name] = data.drop(columns="class").values
        dat.index = data.index

    print(f"Assert {name} to data, data shape: {dat.shape}")

    if not test_env:
        dat.to_csv(dat_path, index=True)

    if test_env:
        print(dat)


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--dat_path", type=str, default="dat/temp_bin.parquet")
    parser.add_argument("--label_type", type=str, default="P_KJX_SH(y=1)")
    parser.add_argument("--weighted_method", type=str, default="equal_weight")
    parser.add_argument("--gen_benchmark", action="store_true")
    parser.add_argument("--get_cumsum", action="store_true")
    parser.add_argument("--get_full", action="store_true")
    parser.add_argument("--test_env", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    former = Portfolio(args.dat_path, args.label_type)
    class_group = former.data.groupby("SMB_HML")
    full_profit = pd.DataFrame()

    for _class in track(CLASSES):
        former.data = class_group.get_group(_class)
        profit = former.form_portfolio(
            args.weighted_method, args.gen_benchmark, args.get_full
        )
        profit.index = profit['date']
        profit.drop(columns='date', inplace=True)
        full_profit = pd.concat([full_profit, profit], axis=0)

    if args.dat_path.split(".")[0].split("_")[-1] == "bin":
        readable_profit = readable_result(full_profit)
        readable_profit = get_hl_portfolio(readable_profit)

        if args.get_cumsum:
            print(readable_profit.cumsum().tail())

        merge_all_label(
            "dat/profit_bin_v2.csv", readable_profit[['HL', 'class']],
            args.label_type, args.weighted_method, args.gen_benchmark, args.test_env
        )

    elif args.dat_path.split(".")[0].split("_")[-1] in ["multi", "mc"]:
        readable_profit = full_profit

        if args.get_cumsum:
            print(readable_profit.cumsum().tail())

        merge_all_label(
            "dat/profit_multi_v2.csv", readable_profit,
            args.label_type, args.weighted_method, args.gen_benchmark, args.test_env
        )

    else:
        raise ValueError("Please specify the label type")
