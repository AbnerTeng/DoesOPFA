"""
Empirical analysis
"""
import os
import warnings
from typing import List

from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from rich.progress import track

from .utils.data_utils import (
    shift_prob,
    clean_binary_prob,
    clean_mc_prob,
    scaling
)
from .utils.metric_utils import Metrics

warnings.filterwarnings("ignore")


class DataManipulate:
    """
    Data Manipulation
    
    * clean_mc_prob function
        ** last argument: accuracy  / portfolio
    """
    def __init__(self, path: str) -> None:
        self.dat = pd.read_parquet(path)

        if self.dat.isna().any().any():
            self.dat.iloc[:, :-1].replace("None", np.nan, inplace=True)

        else:
            self.dat.replace("None", np.nan, inplace=True)

        self.prob_cols = [col for col in self.dat.columns if col.split("_")[0] == "P"]
        self._cls = ['SH', 'SM', 'SL', 'BH', 'BM', 'BL']


    def get_prob(self) -> None:
        """
        get the y=1 probability from the KJX
        """
        for col in track(self.prob_cols):
            if col.split("_")[1] == "KJX":
                clean_binary_prob(self.dat, col, "accuracy")

            elif col.split("_")[1] == "MC":
                clean_mc_prob(self.dat, col, "portfolio")

            else:
                raise ValueError("Invalid column name")


    def col_name_changer(self, mdl_type: str) -> None:
        """
        Change column names
        """
        self.dat.drop(
            columns=['bm', 'SMB', 'HML', 'VOL'],
            inplace=True
        )

        for i in self._cls:
            for j in self._cls:
                if i != j:
                    if mdl_type == "bin":
                        self.dat.loc[self.dat['SMB_HML'] == i, f'P_KJX_{j}'] = \
                            self.dat.loc[self.dat[f'P_KJX_{i}_by{j}'].notna(), f'P_KJX_{i}_by{j}']
                    elif mdl_type == "multi":
                        self.dat.loc[self.dat['SMB_HML'] == i, f'P_MC_{j}'] = \
                            self.dat.loc[self.dat[f'P_MC_{i}_by{j}'].notna(), f'P_MC_{i}_by{j}']
                    else:
                        raise ValueError("Invalid model type")

        self._cls.append('ALL')

        if mdl_type == "bin":
            cols = [f"P_KJX_{i}" for i in self._cls]
        elif mdl_type == "multi":
            cols = [f"P_MC_{i}" for i in self._cls]
        else:
            raise ValueError("Invalid model type")

        self.dat = self.dat[
            ['PERMNO', 'date', 'SMB_HML', 'RET', 'mktcap'] + cols
        ]


    def concat_prob(self, get_dtypes: bool, save: bool, mdl_type: str) -> None:
        """
        Concatenate the probability
        """
        nadat = self.dat.dropna()
        for item in track(self._cls):
            if mdl_type == "bin":
                nadat.rename(
                    columns={f"P_KJX_{item}": f"P_KJX_{item}(y=1)"},
                    inplace=True
                )
            elif mdl_type == "multi":
                nadat.rename(
                    columns={f"P_MC_{item}": f"P_MC_{item}(y=max)"},
                    inplace=True
                )
            else:
                raise ValueError("Empty column")


        for col in track(nadat.columns):
            if col.startswith("P_"):
                nadat[col] = shift_prob(nadat, col)

        if get_dtypes:
            print(nadat.dtypes)

        for col in nadat.columns:
            if "P_" in col:
                if isinstance(nadat[col].iloc[1], str):
                    nadat[col] = nadat[col].astype(float)
            elif col in ["PERMNO", "date"]:
                nadat[col] = nadat[col].astype(float)

        nadat.reset_index(drop=True, inplace=True)
        if not save:
            print(nadat)
        if save:
            nadat.to_parquet(f"dat/mdl_output_{mdl_type}_portfolio.parquet")


class Analysis:
    """
    Empirical analysis
    """
    def __init__(self, prob_df: pd.DataFrame, cls_lst: List[str]) -> None:
        self.prob_df = prob_df
        self.used_cols = [col for col in self.prob_df.columns if "P_" in col]
        self.uniq_ym = list(self.prob_df["date"].unique())
        self._init_metric()
        self.cls_lst = cls_lst


    def _init_metric(self) -> None:
        self.bin_metric = {
            "ym": self.uniq_ym,
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f_score": []
        }
        self.multi_metric = {
            "ym": self.uniq_ym,
            "accuracy": [],
            "entropy": [],
            "precision": [],
            "recall": [],
            "f_score": []
        }


    def binary_metric(self, col: str, target: int) -> None:
        """
        y=1 accuracy for binary classification (KJX)
        
        scaling: Min-Max Scaling to prevent all 0
        
        After scaling, the threshold will be the mean of the scaled value
        """
        self.prob_df["RET"] = self.prob_df["RET"].apply(lambda x: 1 if float(x) > 0 else 0)
        self.prob_df[col] = scaling(self.prob_df[col], "minmax")
        threshold = self.prob_df[col].median()
        self.prob_df[col] = self.prob_df[col].apply(lambda x: 1 if x > threshold else 0)

        for date in self.uniq_ym:
            date_df = self.prob_df[self.prob_df["date"] == date]
            met = Metrics(date_df[col], date_df["RET"])
            self.bin_metric["accuracy"].append(met.accuracy())
            self.bin_metric["precision"].append(met.precision(target))
            self.bin_metric["recall"].append(met.recall(target))
            self.bin_metric["f_score"].append(met.f_score(target))
        metric_df = pd.DataFrame(self.bin_metric)
        return metric_df


    def multi_acc(self, col: str) -> None:
        """
        y=H/L accuracy for multi classification (MC)
        """
        for date in self.uniq_ym:
            date_df = self.prob_df[self.prob_df["date"] == date]
            met = Metrics(date_df[col], date_df['label'])
            self.multi_metric["accuracy"].append(met.accuracy())
            self.multi_metric["entropy"].append(met.entropy(10))
            self.multi_metric["precision"].append(met.precision(9))
            self.multi_metric["recall"].append(met.recall(9))
            self.multi_metric["f_score"].append(met.f_score(9))
        metric_df = pd.DataFrame(self.multi_metric)
        return metric_df


    def reset_dict(self):
        """
        reset the self.metric
        """
        self._init_metric()


def parse_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="metric")
    parser.add_argument("--get_dtypes", action="store_true")
    parser.add_argument("--part", type=str, default="bin")
    parser.add_argument("--prob_data", type=str, default="temp_bin.parquet")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--test_env", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    CLASSES = ["SH", "SM", "SL", "BH", "BM", "BL"]
    args = parse_args()
    if args.mode == "format":
        if (
            "mdl_output_multi_portfolio.parquet" not in os.listdir("dat")
        ) or (
            "mdl_output_bin.parquet" not in os.listdir("dat")
        ):
            if args.part == "bin":
                ORG_PATH = "dat/monthly_data_bin.parquet"

            elif args.part == "multi":
                ORG_PATH = "dat/monthly_data_mc.parquet"

            else:
                raise ValueError("Invalid part name")

            manipulate = DataManipulate(ORG_PATH)
            manipulate.get_prob()
            manipulate.col_name_changer(args.part)
            manipulate.concat_prob(args.get_dtypes, args.save, args.part)

    elif args.mode == "metric":
        if (args.part == "bin") and (args.prob_data in os.listdir("dat")):
            full_kjx_metric = pd.DataFrame()
            bin_dat = pd.read_parquet(f"dat/{args.prob_data}")

            if "date" not in bin_dat.columns:
                bin_dat.rename(columns={"next_date": "date"}, inplace=True)

            bin_cols = [col for col in bin_dat.columns if "P_KJX" in col]

            if args.test_env:
                bin_dat = bin_dat[
                    ['PERMNO', 'date', 'SMB_HML', 'RET', bin_cols[0], bin_cols[1]]
                ]

            for _class in track(CLASSES):
                sub_bin_dat = bin_dat.groupby("SMB_HML").get_group(_class)
                analysis = Analysis(sub_bin_dat, CLASSES)

                for bi_col in analysis.used_cols:
                    output = analysis.binary_metric(bi_col, args.target)
                    output['model'] = bi_col
                    output['class'] = _class
                    analysis.reset_dict()
                    full_kjx_metric = pd.concat([full_kjx_metric, output], axis=0)

            if args.test_env:
                print(full_kjx_metric)
                print(full_kjx_metric.shape)

            else:
                files = os.listdir("dat")

                for _file in files:
                    if _file.startswith("full_kjx_metric_v"):
                        version = int(_file.split(".")[0].split("_")[-1][-1])
                        full_kjx_metric.to_csv(f"dat/full_kjx_metric_v{version+1}.csv", index=False)

                    else:
                        full_kjx_metric.to_csv("dat/full_kjx_metric_v1.csv", index=False)

        elif (args.part == "multi") and (args.prob_data in os.listdir("dat")):
            full_mc_metric = pd.DataFrame()
            mc_dat = pd.read_parquet(f"dat/{args.prob_data}")

            if "date" not in mc_dat.columns:
                mc_dat.rename(columns={"next_date": "date"}, inplace=True)

            multi_cols = [col for col in mc_dat.columns if "P_MC" in col]
            label_data = pd.read_csv("label/test_label_multi_10_org.csv", index_col=0)

            if isinstance(mc_dat["date"].iloc[0], pd.Timestamp):
                mc_dat["date"] = mc_dat["date"].apply(lambda x: x.strftime("%Y%m"))

            else:
                mc_dat['date'] = mc_dat['date'].astype(int).astype(str)

            mc_dat['fig_name'] = \
                mc_dat['PERMNO'].astype(int).astype(str) + \
                    "_" + mc_dat['date']
            concat_dat = pd.merge(
                label_data, mc_dat,how="inner", on="fig_name"
            ).drop_duplicates(subset=["fig_name"])
            concat_dat_new = concat_dat[(concat_dat[multi_cols] != 0).any(axis=1)]

            if args.test_env:
                concat_dat_new = concat_dat[
                    ['label', 'PERMNO', 'date', 'SMB_HML', 'RET', multi_cols[0]]
                ]

            for _class in track(CLASSES):
                sub_mc_dat = concat_dat_new.groupby("SMB_HML").get_group(_class)
                analysis = Analysis(sub_mc_dat, CLASSES)

                for mc_col in analysis.used_cols:
                    output = analysis.multi_acc(mc_col)
                    output['model'] = mc_col
                    output['class'] = _class
                    analysis.reset_dict()
                    full_mc_metric = pd.concat([full_mc_metric, output], axis=0)

            if args.test_env:
                print(full_mc_metric)
                print(full_mc_metric.shape)

            else:
                files = os.listdir("dat")

                for _file in files:
                    if _file.startswith("full_mc_metric_v"):
                        version = int(_file.split(".")[0].split("_")[-1][-1])
                        full_mc_metric.to_csv(f"dat/full_mc_metric_v{version+1}.csv", index=False)

                    else:
                        full_mc_metric.to_csv("dat/full_mc_metric_v1.csv", index=False)

        else:
            raise ValueError(f"{args.prob_data}: File not found")
