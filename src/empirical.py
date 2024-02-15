# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.data_utils import shift
# %%
dat = pd.read_parquet("../dat/monthly_data.parquet").replace(["None", "nan"], np.nan)

prob_cols = [col for col in dat.columns if col.split("_")[0] == "P"]
for col in tqdm(prob_cols):
    if col.split("_")[1] == "KJX":
        dat.loc[dat[col].notna(), col] = \
            dat.loc[dat[col].notna(), col].apply(
                lambda x: [
                    float(val) for val in x.strip('[]').split()
                ]
            )
        dat.loc[dat[col].notna(), col] = \
            dat.loc[dat[col].notna(), col].apply(
                lambda x: x[1]
            )
    elif col.split("_")[1] == "MC":
        if isinstance(dat[col].iloc[0], list):
            dat.loc[dat[col].notna(), col] = \
                dat.loc[dat[col].notna(), col].apply(
                    lambda x: x[0]
                )
        else:
            pass
# %%
dat.drop(
    columns=['mktcap', 'bm', 'SMB', 'HML', 'VOL'],
    inplace=True
)

_cls = ['SH', 'SM', 'SL', 'BH', 'BM', 'BL']
for i in _cls:
    for j in _cls:
        if i != j:
            dat.loc[dat['SMB_HML'] == i, f'P_KJX_{j}'] = \
                dat.loc[dat[f'P_KJX_{i}_by{j}'].notna(), f'P_KJX_{i}_by{j}']
# %%

# %%
dat = pd.read_parquet("../dat/mdl_output.parquet")
# %%
## monthly positive discovery rate
_bin = dat.drop(columns=['P_MC_L(y=L)', 'P_MC_H(y=H)'])
p_cols = [col for col in _bin.columns if "P_" in col]
factor = {
    "ym": list(_bin['date'].unique()),
    "pos_ret": [],
}
for ym in tqdm(_bin['date'].unique()):
    sub = _bin[_bin['date'] == ym]
    for col in p_cols:
        if f"{col}_mean" not in factor:
            factor[f"{col}_mean"] = []
        factor[f"{col}_mean"].append(sub[col].mean())
    factor["pos_ret"].append(len(sub[sub['RET'] > 0]) / sub.shape[0])
# %%
factor_df = pd.DataFrame(factor).reset_index(drop=True)
factor_df["ym"] = pd.to_datetime(factor_df["ym"], format="%Y%m")
# %%
plt.figure(figsize=(12, 6))
plt.plot(factor_df.iloc[:, 1:])
plt.legend(factor_df.columns[1:])
plt.show()
# %%
## percentage of correctly predicting the top / low decile
mc = dat[['P_MC_L(y=L)', 'P_MC_H(y=H)']]

