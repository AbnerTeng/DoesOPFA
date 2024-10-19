"""
Useful utils functions
"""
import os
import math
from typing import Dict, Any, Literal, List, Union
from itertools import chain

import yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_config(config_path: str) -> Dict[str, Any]:
    """
    load config files
    """
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


def to_dataframe(arr: np.ndarray, name_list: List[str]) -> pd.DataFrame:
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
    permno, date = data["name"].str.split("_", expand=True).values.T
    data["PERMNO"], data["date"] = permno, date
    data = data.drop(columns=["name"])
    ret_df = ret_df.astype(str)
    ret_df = ret_df.merge(data, how="left", on=["PERMNO", "date"])

    return ret_df


def reset(func):
    """
    Reset the metric after the function
    """
    def wrapper(self, *args, **kwargs):
        initial_metric = self.metric.copy()
        result = func(self, *args, **kwargs)
        self.metric = initial_metric
        return result

    return wrapper


def filter_imgs(img_path: str, label_path: str, _cls: str) -> pd.DataFrame:
    """
    check whether the image is in the dataframe
    """
    fig_name = [fig.split(".")[0] for fig in os.listdir(img_path)]
    label = pd.read_csv(label_path, index_col=0)
    # TODO: Deal with the case while specific ff3 class
    if _cls != "ALL":
        label = label[label["SMB_HML"] == _cls]

    label = label[label["fig_name"].isin(fig_name)]

    return label


def filter_files(img_path: str, label_path: str, _cls: str) -> list:
    """
    list all fig name in the dataframe
    """
    fig_name = [fig.split(".")[0] for fig in os.listdir(img_path)]
    label = pd.read_csv(label_path, index_col=0)
    # TODO: Same here
    if _cls != "ALL":
        label = label[label["SMB_HML"] == _cls]

    fig_name = label[
        label["fig_name"].isin(fig_name)
    ]["fig_name"].values.tolist()

    fig_name = [f"{img_path}{x}.png" for x in fig_name]
    return fig_name


def __getshape__(data: pd.DataFrame) -> tuple:
    return (
        data[data["new_label"] == 0].shape,
        data[data["new_label"] == 1].shape
    )


def shift(data: pd.DataFrame, cols: Union[str, List[str]]) -> list:
    """
    shift the column by one day

    ** The last yyyymm has no label so we replace it with 0 **
    """
    comp_gp = data.groupby("PERMNO")
    shifted_tot = []

    if isinstance(cols, str):
        cols = [cols]

    for comp in comp_gp.groups.keys():
        subsample = comp_gp.get_group(comp)
        shifted = list(subsample[cols][1:]) + [0]
        shifted_tot.append(shifted)

    return list(chain(*shifted_tot))


def shift_prob(data: pd.DataFrame, cols: str) -> list:
    """
    shift the probability by past one month
    """
    shifted_tot = []
    for _, group in data.groupby("PERMNO"):
        shifted = [0] + list(group[cols][:-1])
        shifted_tot.append(shifted)
    return list(chain(*shifted_tot))


def simple_sharpe(ret: pd.Series, rf: pd.Series) -> float:
    """
    Calculate the simple Sharpe ratio
    """
    if isinstance(ret, pd.Series) | isinstance(rf, pd.Series):
        ret = np.array(ret)
        rf = np.array(rf)

    return ((ret - rf).mean() / ret.std()) * math.sqrt(12)


def calculate_sharpe(data: pd.DataFrame, col: str, rf: float=0.00) -> np.ndarray:
    """
    Calculate Annual Sharpe Ratio

    SR_i = (E(R_i) - R_f) / std(R_i) * sqrt(12)
    """
    if col == "ALL":
        sharpe_df = (
            (data.mean(numeric_only=True) - rf)/ data.std(numeric_only=True)
        ) * np.sqrt(12)
        sharpe_df = sharpe_df.to_frame(name="Annual Sharpe Ratio")
        sharpe_df.to_csv("dat/sharpe_ratio.csv")

    else:
        sharpe = (
            np.mean(data[col] - rf) / (np.std(data[col]) + 1e-8)
        ) * np.sqrt(12)

        print(f"Sharpe Ratio for {col}: {sharpe}")

    return sharpe


def get_alpha(ret_data: pd.DataFrame, ff5_path: str) -> None:
    """
    Construct Fama-French 5 factors model and get alpha
    
    FF5 model
    ---------
    R_it - R_Ft = a_i + b_i(R_Mt - R_Ft) + s_i * SMB_t + h_i * HML_t
                  + r_i * RMW_t + c_i * CMA_t + e_it

    The return is a_i
    """
    ff5_data = pd.read_csv(ff5_path, encoding="utf-8").rename(columns={"Unnamed: 0": "ym"})
    ff5_data = ff5_data[
        (ff5_data["ym"] > 200012) & (ff5_data["ym"] < 202301)
    ].reset_index(drop=True)
    ret_data = ret_data.iloc[:, 1:].reset_index(drop=True)
    feats = ff5_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    feats = sm.add_constant(feats)
    model = sm.OLS(ret_data, feats).fit()
    alpha = pd.DataFrame(
        {
            "class": ret_data.columns,
            "alpha": model.params.iloc[0, :]
        },
    )
    print(alpha)
    alpha.to_csv("dat/alpha.csv")


def clean_binary_prob(
    data: pd.DataFrame,
    label_type: str,
    method: Literal["accuracy", "portfolio"]
) -> pd.DataFrame | None:
    """
    clean the data for binary model
    
    * label_type: str
        ** Example: "P_KJX_SH"

     * method: str
        ** Params: "accuracy" / "portfolio"
        
        ** method = "accuracy" -> get the last probability
        ** method = "portfolio" -> get the label (argmax)
    
    first_prob: the first probability in the column that is not NaN
    """
    first_prob = data.loc[data[label_type].notna(), label_type].iloc[0]

    if isinstance(first_prob, str):
        data.loc[data[label_type].notna(), label_type] = data.loc[
            data[label_type].notna(), label_type
        ].apply(
            lambda x: [float(val) for val in x.strip('[]').split()]
        )

    elif isinstance(first_prob, np.ndarray):
        pass

    else:
        raise ValueError("Invalid probability type")

    if method == "accuracy":
        data.loc[data[label_type].notna(), label_type] = data.loc[
            data[label_type].notna(), label_type
        ].apply(
            lambda x: x[-1]
        )
        return data

    if method == "portfolio":
        return data[label_type].apply(lambda x: x[-1])

    else:
        raise ValueError("Invalid method")


def clean_mc_prob(
    data: pd.DataFrame,
    label_type: str,
    method: Literal["accuracy", "portfolio"]
) -> pd.DataFrame | None:
    """
    clean the data for multi-class model
    
    * label_type: str
        ** Example: "P_MC_SH"

    * method: str
        ** Params: "accuracy" / "portfolio"
        
        ** method = "accuracy" -> get the last probability (y=10)
        ** method = "portfolio" -> get the label (argmax)
    
    first_prob: the first probability in the column that is not NaN
    """
    first_prob = data.loc[data[label_type].notna(), label_type].iloc[0]

    if isinstance(first_prob, str):
        data.loc[data[label_type].notna(), label_type] = data.loc[
            data[label_type].notna(), label_type
        ].apply(
            lambda x: [float(val) for val in x.strip('[]').split()]
        )

    elif isinstance(first_prob, np.ndarray):
        pass

    else:
        raise ValueError("Invalid probability type")

    if method == "accuracy":
        data.loc[data[label_type].notna(), label_type] = data.loc[
            data[label_type].notna(), label_type
        ].apply(
            lambda x: np.argmax(x)
        )
        return data

    if method == "portfolio":
        data[f"{label_type}_H"] = data.loc[
            data[label_type].notna(), label_type
        ].apply(
            lambda x: x[-1]
        )
        data[f"{label_type}_L"] = data.loc[
            data[label_type].notna(), label_type
        ].apply(lambda x: x[0])
        return data

    else:
        raise ValueError("Invalid method")


def scaling(series: pd.Series, method: Literal["minmax", "std"]) -> pd.Series | None:
    """
    Min-Max scaling or Standard scaling
    """
    if isinstance(series, pd.Series):
        output = {
            "minmax": (series - series.min()) / (series.max() - series.min()),
            "std": (series - series.mean()) / series.std()
        }
        return output[method]

    raise TypeError("Input must be a pandas Series")


def ff_processor(path: str) -> pd.DataFrame:
    """
    Process with data downloads from Ken French's website

    Args:
        data (pd.DataFrame): original data downloaded from Ken French's website

    Returns:
        pd.DataFrame: cleaned data
    """
    df = pd.read_csv(path)
    df.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    df.date = df.date.astype(int)
    df = df[(df['date'] > 200012) & (df['date'] < 202301)]
    df.date = pd.to_datetime(df.date, format='%Y%m')
    df.index = df['date']
    df.drop(columns=['date'], inplace=True)
    df = df / 100

    return df
