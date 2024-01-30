import numpy as np
import pandas as pd


def get_ova(arr: np.ndarray, direction: str) -> np.ndarray:
    """
    get the one vs all (OVA) label
    """
    if direction == "L":
        return np.where(arr == 0, 1, 0)
    elif direction == "H":
        return np.where(arr == 10, 1, 0)


def to_dataframe(arr: np.ndarray, name_list: list) -> pd.DataFrame:
    """
    turn the array to dataframe
    """
    data = pd.DataFrame(
        {
            "name": name_list,
            "label": arr
        }
    )
    data["name"] = data["name"].apply(lambda x: x.split("/")[-1].split(".")[0])
    return data


def merge_to_ret_df(data: pd.DataFrame, ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    merge the label to the ret_df based on the same PERMNO and date
    """
    PERMNO, date = data["name"].str.split("_", expand=True).values.T
    data["PERMNO"], data["date"] = PERMNO, date
    data = data.drop(columns=["name"])
    ret_df = ret_df.astype(str)
    ret_df = ret_df.merge(data, how="left", on=["PERMNO", "date"])
    return ret_df
    