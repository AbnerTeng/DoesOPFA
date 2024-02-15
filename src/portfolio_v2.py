"""
Construct portfolio with different strategies
"""
import os
import warnings
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils.data_utils import shift
warnings.filterwarnings("ignore")


class Portfolio:
    """
    Form Portfolio
    """
    def __init__(self, dat_path: str, label_type: str) -> None:
        """
        self.data -> monthly_data.parquet
        
        ...
           PERMNO    date     mktcap     bm   SMB HML  SMB_HML    VOL      RET     P_KJX_SH ...
        0  10001    199301    15120.0  0.604   S   M      SM     44.0   0.000000   [p1, p2]
        1  10001    199302    15390.0  0.691   S   M      SM     23.0   0.017857   [p1, p2]
        2  10001    199303   15318.75  0.691   S   M      SM    137.0   0.011053   [p1, p2]
        3  10001    199304   16393.75  0.691   S   M      SM    473.0   0.070175   [p1, p2]
        4  10001    199305  16259.375  0.753   S   M      SM    126.0  -0.008197   [p1, p2]
        """
        self.data = pd.read_parquet(dat_path).replace(["None", "nan"], np.nan)
        self.label_type = label_type
        self.profit = {}


    def shift_columns(self) -> None:
        """
        shift return and volume by one month
        """
        for col in ['RET', 'VOL', 'mktcap']:
            self.data[f"shifted_{col}"] = shift(self.data, col)
            self.data[f"shifted_{col}"] = pd.to_numeric(
                self.data[col], errors='coerce'
            ).fillna(0).astype(float)


    def form_portfolio(
            self, weighted_method: str, gen_benchmark: bool, get_full: bool
        ) -> pd.DataFrame:
        """
        form portfolio with different strategies
        
        ---
        1. equal_weight
        2. vol_weight
        3. value_weight
        
        ---
        Portfolio:
        * Benchmark:
            Buy and hold all
        * H-L portfolio:
            Long the highest 10% probability of rise, short the lowest 10% probability of rise

        """
        sub_data = self.data[self.data[self.label_type].notna()]
        sub_data = sub_data[
            ['PERMNO', 'date', 'shifted_RET', 'shifted_VOL', 'shifted_mktcap', self.label_type]
        ]
        prob_cols = [col for col in self.data.columns if col.split("_")[0] == "P"]
        if self.label_type != prob_cols[-1]:
            sub_data[self.label_type] = sub_data[self.label_type].apply(
                lambda x: x.strip("[]").split()
            )
        for ym in tqdm(sub_data.date.unique()):
            sub_data_ym = sub_data[sub_data.date == ym]
            if self.label_type.split("_")[1] == "KJX":
                if self.label_type != prob_cols[-1]:
                    sub_data_ym['rise_prob'] = sub_data_ym[self.label_type].apply(
                        lambda x: float(x[1])
                    )
                else:
                    sub_data_ym['rise_prob'] = sub_data_ym[self.label_type].apply(
                        lambda x: x[1]
                    )

                sub_data_ym['label'] = self.gen_decile(sub_data_ym['rise_prob'], 10)
            elif self.label_type.split("_")[1] == "MC":
                if isinstance(sub_data_ym[self.label_type].iloc[0], list):
                    sub_data_ym[self.label_type] = sub_data_ym[self.label_type].apply(
                        lambda x: float(x[0])
                    )
                sub_data_ym['label'] = self.gen_decile(sub_data_ym[self.label_type], 10)
            if gen_benchmark:
                self.profit[ym] = self.get_benchmark(sub_data_ym, weighted_method)
            if get_full:
                self.profit[ym] = self.get_full_decile(sub_data_ym, weighted_method)
            else:
                print("Please specify the return type")

        profit_df = pd.DataFrame(
            self.profit.items(), columns=['date', 'profit']
        )
        profit_df.sort_values(by='date', inplace=True)
        return profit_df


    @staticmethod
    def gen_decile(col: pd.Series, _range: int) -> np.ndarray:
        """
        Generate decile of return prob in specific month based on different strategies
        """
        percentile = np.percentile(col, np.arange(10, 100, _range))
        return np.digitize(col, percentile)


    @staticmethod
    def gen_weight(data: pd.DataFrame, weight_method: str, label: int) -> pd.Series:
        """
        Generate weight based on different strategies
        """
        weight_dict = {
            "equal_weight": np.where(data.label == label, 1 / data.shape[0], 0),
            "value_weight": np.where(
                data.label == label,
                data.shifted_mktcap / sum(data.shifted_mktcap),
                -1 * data.shifted_mktcap / sum(data.shifted_mktcap)
            ),
            "vol_weight": np.where(
                data.label == label,
                data.shifted_VOL / sum(data.shifted_VOL),
                -1 * data.shifted_VOL / sum(data.shifted_VOL)
            )
        }
        return weight_dict[weight_method]


    @staticmethod
    def gen_bench_weight(data: pd.DataFrame, weight_method: str) -> pd.Series:
        """
        Generate weight based on different strategies
        """
        weight_dict = {
            ## TODO: VOL \times PRC
            "equal_weight_bench": 1 / data.shape[0],
            "vol_weight_bench": data.shifted_VOL / sum(data.shifted_VOL),
            "value_weight_bench": data.shifted_mktcap / sum(data.shifted_mktcap)
        }
        return weight_dict[weight_method]


    def get_benchmark(self, data: pd.DataFrame, weight_method: str) -> float:
        """
        Get Benchmark
        
        input arguments
        ---------------
        data: pd.DataFrame
            May be sub_data_ym in this case
        """
        data["weight"] = self.gen_bench_weight(data, weight_method)
        return sum(data['weight'] * data['shifted_RET'])


    def get_full_decile(self, data: pd.DataFrame, weight_method: str) -> list:
        """
        Get full decile return (0-9)
        
        input arguments
        ---------------
        data: pd.DataFrame
            May be sub_data_ym in this case
        """
        decile_lst = []
        for label in range(10):
            data["weight"] = self.gen_weight(data, weight_method, label)
            decile_lst.append(sum(data['weight'] * data['shifted_RET']))
        return decile_lst


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
    return output


def get_hl_portfolio(data: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """
    Get H-L portfolio, ret = decile_10 - decile_1
    """
    if label_type.split("_")[2] == "H":
        data["HL"] = data["decile_10"] - data["decile_1"]
    elif label_type.split("_")[2] == "L":
        data["HL_inv"] = data["decile_1"] - data["decile_10"]
    return data


def merge_all_label(
        dat_path: str, series: pd.Series, label: str, weight: str, gen_benchmark: bool
    ) -> None:
    """
    Merge all cum_profit
    """
    if not os.path.exists(dat_path):
        dat = pd.DataFrame()
    else:
        dat = pd.read_csv(dat_path, index_col=0)
    name = f"{label}_{weight}_bench" if gen_benchmark else f"{label}_{weight}"
    dat[name] = series
    print(f"Assert {name} to data, shape: {dat.shape}")
    dat.to_csv(dat_path, index=True)


def parse_args() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--dat_path", type=str, default="dat/monthly_data.parquet")
    parser.add_argument("--label_type", type=str, default="P_KJX_SH")
    parser.add_argument("--weighted_method", type=str, default="equal_weight")
    parser.add_argument("--gen_benchmark", action="store_true")
    parser.add_argument("--get_cumsum", action="store_true")
    parser.add_argument("--get_full", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    former = Portfolio(args.dat_path, args.label_type)
    former.shift_columns()
    profit = former.form_portfolio(args.weighted_method, args.gen_benchmark, args.get_full)
    profit['date'] = pd.to_datetime(profit['date'], format="%Y%m")
    profit.index = profit['date']
    profit.drop(columns='date', inplace=True)
    profit = readable_result(profit)
    profit = get_hl_portfolio(profit, args.label_type)
    print(profit.head())
    if args.get_cumsum:
        print(profit.cumsum().tail())
    merge_all_label(
        "dat/every_profit.csv", profit[profit.columns[-1]],
        args.label_type, args.weighted_method, args.gen_benchmark
    )
