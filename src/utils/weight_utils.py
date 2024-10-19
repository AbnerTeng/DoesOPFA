"""
weighted and decile generation functions
"""
from typing import List, Literal, Union


import numpy as np
import pandas as pd


def gen_decile(col: pd.Series, _range: int) -> np.ndarray:
    """
    Generate decile of return prob in specific month based on different strategies
    """
    percentile = np.percentile(col, np.arange(10, 100, _range))
    return np.digitize(col, percentile)


def gen_weight(
    data: pd.DataFrame,
    weight_method: Literal["equal_weight", "value_weight"],
    label: int
) -> np.ndarray:
    """
    Generate weight based on different strategies

    Args:
        label (int): the class label
        - Multiclass in [0, 9]
        - Binary in [0, 1]
    """
    target = data[data.label == label]
    weight_dict = {
        "equal_weight": np.where(
            data.label == label, 1 / (target.shape[0] + 1), 0
        ),
        "value_weight": np.where(
            data.label == label,
            data.mktcap / (sum(target.mktcap) + 1), 0
        )
    }
    return weight_dict[weight_method]


def gen_bench_weight(
    data: pd.DataFrame,
    weight_method: Literal["equal_weight_bench", "value_weight_bench"]
) -> Union[np.ndarray, float]:
    """
    Generate weight based on different strategies
    """
    weight_dict = {
        "equal_weight_bench": 1 / data.shape[0],
        "value_weight_bench": data.mktcap / sum(data.mktcap)
    }
    return weight_dict[weight_method]


def get_benchmark(
    data: pd.DataFrame,
    weight_method: Literal["equal_weight_bench", "value_weight_bench"]
) -> float:
    """
    Get Benchmark
    
    input arguments
    ---------------
    data: pd.DataFrame
        May be sub_data_ym in this case
    """
    data["weight"] = gen_bench_weight(data, weight_method)
    return sum(data['weight'] * data['RET'])


def get_full_decile(
    data: pd.DataFrame,
    weight_method: Literal["equal_weight", "value_weight"]
) -> List[float]:
    """
    Get full decile return (0-9)
    
    input arguments
    ---------------
    data: pd.DataFrame
        May be sub_data_ym in this case
    """
    decile_lst = []

    for label in range(10):
        data["weight"] = gen_weight(data, weight_method, label)
        decile_lst.append(sum(data['weight'] * data['RET']))

    return decile_lst


def high_minus_low(
    data: pd.DataFrame,
    weight_method: Literal["equal_weight", "value_weight"]
) -> List[float]:
    """
    Get the high minus low return
    
    In multi-class classification, we only record the
    highest label (y=9) and the lowest label (y=0)
    """
    data["weight_H"] = gen_weight(data, weight_method, 9)
    data["weight_L"] = gen_weight(data, weight_method, 0)

    return sum(data['weight_H'] * data['RET']) - sum(data['weight_L'] * data['RET'])
