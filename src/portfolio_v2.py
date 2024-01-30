"""
Construct portfolio with different strategies
"""
import warnings
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import shift
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
        self.data = pd.read_parquet(dat_path)
        self.data.replace("None", np.nan, inplace=True)
        self.label_type = label_type
        self.profit = dict()


    def shift_columns(self) -> None:
        """
        shift return and volume by one month
        """
        # for col in ['RET', 'VOL', 'mktcap']:
        #     self.data[f"shifted_{col}"] = shift(self.data, col)
        #     self.data[f"shifted_{col}"] = pd.to_numeric(
        #         self.data[col], errors='coerce'
        #     ).fillna(0).astype(float)
        self.data['shifted_ret'] = shift(self.data, 'RET')
        self.data['shifted_ret'] = pd.to_numeric(
            self.data['shifted_ret'], errors='coerce'
        ).fillna(0).astype(float)
        self.data['shifted_vol'] = shift(self.data, 'VOL')
        self.data['shifted_vol'] = pd.to_numeric(
            self.data['shifted_vol'], errors='coerce'
        ).fillna(0).astype(float)
        self.data['shifted_mktcap'] = shift(self.data, 'mktcap')
        self.data['shifted_mktcap'] = pd.to_numeric(
            self.data['shifted_mktcap'], errors='coerce'
        ).fillna(0).astype(float)


    def form_portfolio(self, weighted_method: str, gen_benchmark: bool) -> pd.DataFrame:
        """
        form portfolio with different strategies
        
        ---
        1. equal_weight
        2. vol_weight
        3. value_weight
        
        ---
        Portfolio:
        * Benchmark: Buy and hold all
        * H-L portfolio: Long the highest 10% probability of rise, short the lowest 10% probability of rise

        """
        if self.label_type.split("_")[1] == "KJX":
            sub_data = self.data[self.data[self.label_type].notna()]
            sub_data = sub_data[
                ['PERMNO', 'date', 'shifted_ret', 'shifted_vol', 'shifted_mktcap', self.label_type]
            ]
            prob_cols = [col for col in self.data.columns if col.split("_")[0] == "P"]
            print(self.label_type, prob_cols[-1])
            if self.label_type != prob_cols[-1]:
                sub_data[self.label_type] = sub_data[self.label_type].str.replace(r"\s+", ", ")
            for ym in tqdm(sub_data.date.unique()):
                sub_data_ym = sub_data[sub_data.date == ym]
                if self.label_type != prob_cols[-1]:
                    sub_data_ym['rise_prob'] = sub_data_ym[self.label_type].apply(
                        lambda x: np.array(eval(x))[1]
                    )
                else:
                    sub_data_ym['rise_prob'] = sub_data_ym[self.label_type].apply(
                        lambda x: x[1]
                    )
                percentile = np.percentile(sub_data_ym['rise_prob'], np.arange(10, 100, 10))
                sub_data_ym['label'] = np.digitize(sub_data_ym['rise_prob'], percentile)
                sub_data_ym['weight'] = self.gen_weight(sub_data_ym, weighted_method, gen_benchmark)
                self.profit[ym] = sum(sub_data_ym['weight'] * sub_data_ym['shifted_ret'])
            profit_df = pd.DataFrame(
                self.profit.items(), columns=['date', 'profit']
            )
            profit_df.sort_values(by='date', inplace=True)
            profit_df['cum_profit'] = profit_df['profit'].cumsum()
            return profit_df


    @staticmethod
    def gen_weight(data: pd.DataFrame, weight_method: str, gen_benchmark: bool) -> pd.Series:
        """
        equal weight portfolio
        """
        benchmark = "bench" if gen_benchmark else "notbench"
        weight_dict = {
            "equal_weight_notbench": np.where(
                data.label == 9, 1 / data.shape[0], -1 / data.shape[0]
            ),
            "vol_weight_notbench": np.where(
                data.label == 9, data.shifted_vol / sum(data.shifted_vol), -1 * data.shifted_vol / sum(data.shifted_vol)
            ),
            "value_weight_notbench": np.where(
                data.label == 9, data.shifted_mktcap / sum(data.shifted_mktcap), -1 * data.shifted_mktcap / sum(data.shifted_mktcap)
            ),
            "equal_weight_bench": 1 / data.shape[0],
            "vol_weight_bench": data.shifted_vol / sum(data.shifted_vol),
            "value_weight_bench": data.shifted_mktcap / sum(data.shifted_mktcap)
        }
        return weight_dict[f"{weight_method}_{benchmark}"]


def merge_all_label(
        dat_path: str, series: pd.Series, label: str, weight: str, gen_benchmark: bool
    ) -> None:
    """
    Merge all cum_profit
    """
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
    parser.add_argument(
        "--dat_path", type=str, default="dat/monthly_data.parquet"
    )
    parser.add_argument(
        "--label_type", type=str, default="P_KJX_SH"
    )
    parser.add_argument(
        "--weighted_method", type=str, default="equal_weight"
    )
    parser.add_argument("--gen_benchmark", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    former = Portfolio(args.dat_path, args.label_type)
    former.shift_columns()
    profit = former.form_portfolio(args.weighted_method, args.gen_benchmark)
    profit['date'] = pd.to_datetime(profit['date'], format="%Y%m")
    profit.index = profit['date']
    profit.drop(columns=['date', 'profit'], inplace=True)
    merge_all_label(
        "dat/every_data.csv", profit['cum_profit'],
        args.label_type, args.weighted_method, args.gen_benchmark
    )
