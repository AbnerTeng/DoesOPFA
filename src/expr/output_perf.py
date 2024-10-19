"""
Output pool of empirical results

Including Annual Sharpe Ratio, Monthly Alpha, Accuracy, etc.
"""
import warnings
from typing import Tuple

from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
import statsmodels.api as sm
from rich.progress import track

from ..utils.data_utils import (
    simple_sharpe,
    ff_processor
)
warnings.filterwarnings("ignore")


class EmpResAcc:
    """
    A bunch of empirical results
    """
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.model_group = data.groupby('model')
        self.mean_accuracy = self.model_group.accuracy.mean()

    @staticmethod
    def fivyavg(df: pd.DataFrame) -> pd.DataFrame:
        """
        Five years average accuracy
        """
        if isinstance(df.ym.iloc[0], np.int64):
            df['ym'] = df['ym'].astype(str)

        df['y'] = df['ym'].apply(lambda x: x[:4])
        y_start, y_end = int(df['y'].min()), int(df['y'].max())
        acc_lst = []

        for _ in range(y_start, y_end, 5):
            sub_df = df[
                (df['y'].astype(int) >= y_start) & (df['y'].astype(int) < y_start + 5)
            ]
            acc_lst.append(sub_df['accuracy'].mean())
            y_start += 5

        acc_df = pd.DataFrame(acc_lst).reset_index(drop=True)
        new_index = [f"{y}-{y+4}" for y in range(2001, 2017, 5)]
        new_index.append('2021-2022')
        acc_df.index = new_index

        return acc_df

    def full_acc(self, to_latex: bool) -> None:
        """
        get full accuracy
        """
        full_df = pd.DataFrame()

        for mdl in self.model_group.groups.keys():
            sub_df = self.model_group.get_group(mdl)
            acc_df = self.fivyavg(sub_df)
            full_df = pd.concat([full_df, acc_df], axis=1)
            full_df.rename(columns={0: mdl}, inplace=True)

        full_df = full_df.apply(lambda x: round(x, 3))

        print(full_df.to_latex() if to_latex else full_df)


class EmpResPerformance:
    """
    A bunch of empirical results ver2
    """
    def __init__(
        self,
        data: pd.DataFrame,
        group: pd.core.groupby.generic.DataFrameGroupBy,
        feature: pd.DataFrame,
        risk_free: pd.Series
    ) -> None:
        self.data = data
        self.group = group
        self.feature = feature
        self.risk_free = risk_free

    def alpha(self, profit: pd.Series, factor: pd.DataFrame) -> Tuple[float, float]:
        """
        Alpha
        """
        if isinstance(profit.index[0], str):
            profit.index = pd.to_datetime(profit.index)

        result = sm.OLS(profit, factor).fit(cov_type="HAC", cov_kwds={'maxlags': 1})
        return result.params[0], result.pvalues[0]

    def get_output(self) -> pd.DataFrame:
        """
        get comprehensive output
        """
        output = {
            'class': [],
            'model': [],
            'SR': [],
            'Alpha': [],
            'p_value': [],
            'cumret': []
        }
        for _class in track(self.group.groups.keys()):
            sub_df = self.group.get_group(_class)
            sub_df = sub_df.drop(columns="class")
            self.feature = self.feature.loc[sub_df.index]
            self.risk_free = self.risk_free.loc[sub_df.index]

            for col in sub_df.columns:
                output['SR'].append(simple_sharpe(sub_df[col], self.risk_free))
                output['Alpha'].append(self.alpha(sub_df[col], self.feature)[0])
                output['p_value'].append(self.alpha(sub_df[col], self.feature)[1])
                output['class'].append(_class)
                output['model'].append(col)
                output['cumret'].append(sub_df[col].cumsum()[-1])

        return pd.DataFrame(output)


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser(description="Output pool of empirical results")
    parser.add_argument("--type", type=str)
    parser.add_argument("--label", type=str)
    parser.add_argument("--metric_data", type=str)
    parser.add_argument("--profit_data", type=str)
    parser.add_argument("--with_bench", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--to_latex", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.type == "acc":
        accuracy_df = pd.read_csv(args.metric_data)
        res_acc = EmpResAcc(accuracy_df)
        res_acc.full_acc(args.to_latex)

    elif args.type == "indicator":
        profit_series = pd.read_csv(args.profit_data, index_col=0)
        group_profit = profit_series.groupby("class")
        ff_data = ff_processor('dat/ff5_dat.csv')
        rf = ff_data['RF']

        if not args.with_bench:
            feats = ff_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]

        else:
            mom_df = ff_processor('dat/MOM.csv')
            str_df = ff_processor('dat/STR.csv')
            ltr_df = ff_processor('dat/LTR.csv')
            feats = pd.concat([ff_data, mom_df, str_df, ltr_df], axis=1)

        feats = sm.add_constant(feats)
        performance = EmpResPerformance(
            profit_series, group_profit, feats, rf
        )
        indicator_perf = performance.get_output()

        if args.save:
            indicator_perf.to_csv(f"dat/indicator_perf_{args.label}.csv")

        else:
            print(indicator_perf)

    else:
        raise ValueError("Invalid type")
