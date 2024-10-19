"""Benchmark Analysis

Returns:
    pd.DataFrame: benchmark return
"""
import os
from typing import Tuple, List

from argparse import ArgumentParser, Namespace
import pandas as pd

from ..constants import FF5_PATH
from ..utils.data_utils import (
    simple_sharpe,
    ff_processor
)


def get_hl(df: pd.DataFrame) -> pd.DataFrame:
    """
    High Prior - Low Prior
    """
    df['HL'] = df['Hi PRIOR'] - df['Lo PRIOR']
    return df


def ret_n_sr(
    bench: pd.DataFrame,
    risk_free: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the return and Sharpe ratio of the benchmark
    """
    ret, sr = [], []
    for col in bench.columns:
        ret.append(bench[col].mean())
        sr.append(simple_sharpe(bench[col], risk_free))

    return ret, sr


def update_data(
    ret: List[float],
    sr: List[float],
    bench: str,
    weight: str
) -> None:
    """
    Update benchmark performance to decile table
    """
    if f"mc_table_ret_{weight}.csv" in os.listdir('dat'):
        ret_df = pd.read_csv(
            f"dat/mc_table_ret_{weight}.csv",
            index_col=0
        )
    if f"mc_table_sr_{weight}.csv" in os.listdir('dat'):
        sr_df = pd.read_csv(
            f"dat/mc_table_sr_{weight}.csv",
            index_col=0
        )
    else:
        raise ValueError("Please generate the decile table first")

    ret_df[bench] = ret
    sr_df[bench] = sr
    ret_df.to_csv(f"dat/mc_table_ret_{weight}.csv", index=True)
    sr_df.to_csv(f"dat/mc_table_sr_{weight}.csv", index=True)


def parse_args() -> Namespace:
    """
    Parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--bench_path", type=str,
        help="benchmark data path"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bench_df = ff_processor(args.bench_path)
    bench_df = get_hl(bench_df)
    bench_type = args.bench_path.split("/")[-1].split("_")[0]
    weight_type = args.bench_path.split("/")[-1].split(".")[0].split("_")[-1]
    ff5_df = ff_processor(FF5_PATH)
    ret_list, sr_list = ret_n_sr(bench_df, ff5_df['RF'])
    update_data(ret_list, sr_list, bench_type, weight_type)
