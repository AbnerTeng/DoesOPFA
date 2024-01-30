"""
Useful utils functions
"""
import os
from itertools import chain
import numpy as np
import pandas as pd
import statsmodels.api as sm


def filter_imgs(img_path: str, label_path: str, _cls: str) -> pd.DataFrame:
    """
    check whether the image is in the dataframe
    """
    fig_name = []
    for fig in os.listdir(img_path):
        fig_name.append(fig.split(".")[0])

    label = pd.read_csv(label_path, index_col=0)
    ## TODO: Deal with the case while specific ff3 class
    if _cls != "ALL":
        label = label[label["SMB_HML"] == _cls]
    label = label[label["fig_name"].isin(fig_name)]
    return label


def filter_files(img_path: str, label_path: str, _cls: str) -> list:
    """
    list all fig name in the dataframe
    """
    fig_name = []
    for fig in os.listdir(img_path):
        fig_name.append(fig.split(".")[0])

    label = pd.read_csv(label_path, index_col=0)
    ## TODO: Same here
    if _cls != "ALL":
        label = label[label["SMB_HML"] == _cls]
    fig_name = label[label["fig_name"].isin(fig_name)]["fig_name"].values.tolist()
    fig_name = [f"{img_path}{x}.png" for x in fig_name]
    return fig_name


def __getshape__(data: pd.DataFrame) -> tuple:
    return data[data["new_label"] == 0].shape, data[data["new_label"] == 1].shape


def shift(data: pd.DataFrame, cols: str) -> list:
    """
    shift the column by one day
    
    ** The last yyyymm has no label so we replace it with 0 **
    """
    comp_gp = data.groupby("PERMNO")
    shifted_tot = []
    for comp in comp_gp.groups.keys():
        subsample = comp_gp.get_group(comp)
        shifted = list(subsample[cols][1:]) + [0]
        shifted_tot.append(shifted)
    return list(chain(*shifted_tot))


def calculate_sharpe(path: str, rf: float=0.00) -> None:
    """
    Calculate Annual Sharpe Ratio
    
    SR_i = (E(R_i) - R_f) / std(R_i) * sqrt(12)
    """
    data = pd.read_csv(path, encoding="utf-8")
    ret_dat = data.iloc[:, 1:].diff().iloc[1:-1, :]
    sharpe_df = ((ret_dat.mean() - rf)/ ret_dat.std()) * np.sqrt(12)
    sharpe_df = sharpe_df.to_frame(name="Annual Sharpe Ratio")
    sharpe_df.to_csv("dat/sharpe_ratio.csv")


def get_alpha(ret_path: str, ff5_path: str) -> None:
    """
    Construct Fama-French 5 factors model and get alpha
    
    FF5 model
    ---------
    R_it - R_Ft = a_i + b_i(R_Mt - R_Ft) + s_i * SMB_t + h_i * HML_t
                  + r_i * RMW_t + c_i * CMA_t + e_it
                  
    The return is a_i
    """
    ret_data = pd.read_csv(ret_path, encoding="utf-8")
    ff5_data = pd.read_csv(ff5_path, encoding="utf-8").rename(columns={"Unnamed: 0": "ym"})
    ff5_data = ff5_data[
        (ff5_data["ym"] > 200101) & (ff5_data["ym"] < 202212)
    ].reset_index(drop=True)
    rt = ret_data.iloc[:, 1:].diff().iloc[1:-1, :].reset_index(drop=True)
    feats = ff5_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    feats = sm.add_constant(feats)
    model = sm.OLS(rt, feats).fit()
    alpha = model.params.iloc[0, :]
    alpha.to_csv("dat/alpha.csv")
