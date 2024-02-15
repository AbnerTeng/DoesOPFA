# %%
import pandas as pd
from tqdm import tqdm
# %%
df = pd.read_csv('../dat/crsp_stock.csv')
df.drop(
    columns="SHRCD",
    inplace=True
)
df.rename(
    columns={
        "BIDLO": "Low",
        "ASKHI": "High",
        "PRC": "Close",
        "VOL": "Volume",
        "RET": "Return",
        "OPENPRC": "Open",
    },
    inplace=True
)
df.Return = pd.to_numeric(df.Return, errors='coerce').fillna(0).astype(float)
df.date = df.date.astype(str).apply(lambda x: x[:6])
df.to_parquet('../dat/newdata_v2.parquet')
# %%
k = 20
df = pd.read_parquet('../dat/newdata_v2.parquet')
grp_comp = df.groupby('PERMNO')
df[f'{k}_MA'] = grp_comp['Close'].transform(lambda x: x.rolling(k).mean())
unique_date = df['date'].unique()
ym_grp = df.groupby('date')
# %%
## TODO book value missing alot
def conditions(prc_df: pd.DataFrame) -> bool:
    """
    the condition of filtering the company
    """
    return not (
        prc_df.isnull().any().any() \
        or min(prc_df['Close']) < 0 \
        or min(prc_df['Volume']) == 0
    )


def get_numfirms(uni_date, grp) -> list:
    """
    get number of firms pass the conditions
    """
    num_firms = []
    for ym in tqdm(uni_date):
        ym_df = grp.get_group(ym)
        num_firms.append(
            ym_df.groupby('PERMNO').apply(
                lambda comp_df: conditions(comp_df)
            ).sum()
        )
    return num_firms


def get_permno(uni_date, grp) -> list:
    """
    get permno of fimrms pass the conditions
    """
    permno = []
    for ym in tqdm(uni_date):
        ym_df = grp.get_group(ym)
        condition_df = ym_df.groupby('PERMNO').apply(
            lambda comp_df: conditions(comp_df)
        )
        permno.append(
            condition_df[condition_df].index.tolist()
        )
    return permno
# %%
# num_firms_data = pd.DataFrame(
#     {
#         'date': unique_date,
#         'num_firms': get_numfirms(unique_date, ym_grp)
#     }
# )
permno_data = pd.DataFrame(
    {
        'date': unique_date,
        'permno': get_permno(unique_date, ym_grp)
    }
)
# permno_data.to_csv('../dat/filtered_permno.csv', index=False)
# %%
## TODO: Filter the full data with filtered permno
final_df = pd.DataFrame()
for ym in tqdm(unique_date):
    sub_df = ym_grp.get_group(ym)
    valid_permno = permno_data[permno_data['date'] == ym]['permno'].values[0]
    permno_df = sub_df[sub_df['PERMNO'].isin(valid_permno)]
    final_df = pd.concat([final_df, permno_df])
# %%
final_df.to_parquet('../dat/filtered_prc.parquet')
# %%
import pandas as pd
bm_ratio = pd.read_parquet('../dat/bm_ratio.parquet')
bm_ratio.rename(
    columns={
        "permno": "PERMNO",
        "public_date": "date",
    },
    inplace=True
)
mktcap = pd.read_parquet('../dat/monthly_mktcap.parquet')
mktcap['mktcap'] = abs(mktcap['PRC']) * mktcap['SHROUT']
bm_ratio.sort_values(by='PERMNO', inplace=True)
merged_df = pd.merge(
    bm_ratio, mktcap,
    how="left", on=['PERMNO', 'date'],
).sort_values(by=['PERMNO', 'date'])
merged_df['book_value'] = merged_df['bm'] * merged_df['mktcap']
merged_df['date'] = merged_df['date'].astype(str).apply(lambda x: x[:6])
# %%
import numpy as np
import pandas as pd
data = pd.read_parquet("../dat/filtered_prc.parquet")
ash_map = data.groupby('PERMNO')['date'].agg(lambda x: np.unique(x)).to_dict()
array = np.array(data[['PERMNO', 'date', 'Open', 'High', 'Low', 'Close', '20_MA', 'Volume']])
hash_map = {k: ash_map[k] for k in list(ash_map.keys())[:2]}
import matplotlib.pyplot as plt

length = []
for k in ash_map.keys():
    length.append(len(ash_map[k]))

data = pd.DataFrame(
    {
        'PERMNO': list(ash_map.keys()),
        'length': length
    }
)
data.index = data['PERMNO']
plt.figure(figsize=(20, 10))
plt.scatter(x=data.index, y=data['length'])
plt.show()
# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
bm_ratio = pd.read_parquet('../dat/bm_ratio.parquet')
mktcap = pd.read_parquet('../dat/monthly_mktcap.parquet')
bm_ratio.rename(
    columns={
        "permno": "PERMNO",
        "public_date": "date",
    },
    inplace=True
)
mktcap['mktcap'] = abs(mktcap['PRC']) * mktcap['SHROUT']
bm_ratio['date'] = bm_ratio['date'].astype(str).apply(lambda x: x[:6])
mktcap['date'] = mktcap['date'].astype(str).apply(lambda x: x[:6])
bm_ratio.sort_values(by=['date', 'PERMNO'], inplace=True)
mktcap.sort_values(by=['date', 'PERMNO'], inplace=True)
bm_ratio.dropna(inplace=True)
mktcap.dropna(inplace=True)
# %%
bm_unique_date = bm_ratio['date'].unique()
cap_unique_date = mktcap['date'].unique()

smb, hml = [], []
for date in tqdm(bm_unique_date):
    sub_bm_df = bm_ratio[bm_ratio['date'] == date]
    bm_quantile = sub_bm_df['bm'].quantile([0, 0.3, 0.7, 1], interpolation='linear')
    hml_series= pd.Series(
        np.where(
            sub_bm_df['bm'] <= bm_quantile.loc[0.3], 'L',
            np.where(
                sub_bm_df['bm'] >= bm_quantile.loc[0.7], 'H', 'M'
            )
        ),
        name='HML'
    )
    hml.append(hml_series)

for date in tqdm(cap_unique_date):
    sub_cap_df = mktcap[mktcap['date'] == date]
    cap_quantile = sub_cap_df['mktcap'].quantile([0, 0.5, 1], interpolation='linear')
    smb_series = pd.Series(
        np.where(
            sub_cap_df['mktcap'] <= cap_quantile.loc[0.5], 'S', 'B'
        ),
        name='SMB'
    )
    smb.append(smb_series)
# %%
hml_arr = np.concatenate(hml)
smb_arr = np.concatenate(smb)
bm_ratio['HML'] = hml_arr
mktcap['SMB'] = smb_arr
# %%
merged_df = pd.merge(
    mktcap, bm_ratio,
    how="inner", on=['PERMNO', 'date'],
).sort_values(by=['PERMNO', 'date'])
merged_df.drop(
    columns=[
        'SHRCD', 'TICKER_x', 'PRC', 'SHROUT', 'adate', 'qdate', 'TICKER_y'
    ],
    inplace=True
)
merged_df['SMB_HML'] = merged_df['SMB'] + merged_df['HML']
merged_df['fig_name'] = merged_df['PERMNO'].astype(str) + '_' + merged_df['date'].astype(str)
merged_df = merged_df[['fig_name', 'mktcap', 'bm', 'SMB', 'HML', 'SMB_HML']]
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(figsize=(20, 10))
for label in merged_df.SMB_HML.unique():
    avg_smb = merged_df[merged_df['SMB_HML'] == label].groupby('date')['bm'].mean()
    axs.plot(merged_df['date'].unique(), avg_smb, label=label)
axs.legend()
plt.show()

# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(figsize=(20, 10))
for label in merged_df.SMB_HML.unique():
    avg_hml = merged_df[merged_df['SMB_HML'] == label].groupby('date')['mktcap'].mean()
    axs.plot(merged_df['date'].unique(), avg_hml, label=label)
axs.legend()
plt.show()
# %%
merged_df.to_parquet('../dat/ff3_class.parquet')
# %%
mth = pd.read_csv("../dat/monthly_data.csv")
ff3 = pd.read_parquet("../dat/ff3_class.parquet")
# %%
# mth['fig_name'] = mth['PERMNO'].astype(str) + '_' + mth['date'].astype(str).apply(lambda x: x[:6])
mth = mth[['PEMRNO', 'date', 'VOL', 'RET']]
# %%
full_data = pd.merge(
    ff3, mth,
    how="inner", on='fig_name'
)
# %%
full_data.to_parquet('../dat/monthly_data.parquet')
# %%
full_data = pd.read_parquet('../dat/monthly_data.parquet')
# %%
full_data['PERMNO'] = full_data['fig_name'].apply(lambda x: x.split('_')[0])
full_data['date'] = full_data['fig_name'].apply(lambda x: x.split('_')[1])

# %%
full_data = full_data[['PERMNO', 'date', 'mktcap', 'bm', 'SMB', 'HML', 'SMB_HML', 'VOL', 'RET']]
# %%
full_data.isnull().sum()
# %%
full_data.to_parquet('../dat/monthly_data.parquet')
# %%
import pandas as pd
import numpy as np
from utils import shift
# %%
data = pd.read_parquet("../dat/monthly_data.parquet")
# %%
data['shifted_ret'] = shift(data, 'RET')
data['shifted_ret'] = pd.to_numeric(data['shifted_ret'], errors='coerce').fillna(0).astype(float)
# %%
sub_data = data[data.P_KJX_SH.notna()]
for ym in sorted(sub_data.date.unique())[:-1]:
    sub_data_ym = sub_data[sub_data['date'] == ym]
    
# %%
sub_data_ym['rise_prob'] = sub_data_ym.P_KJX_SH.apply(lambda x: x[1])
percentile = np.percentile(sub_data_ym['rise_prob'], np.arange(10, 100, 10))
sub_data_ym['label'] = np.digitize(sub_data_ym['rise_prob'], percentile)
# %%
sub_data_ym = sub_data_ym[(sub_data_ym['label'] == 9) | (sub_data_ym['label'] == 0)]
# %%
sub_data_ym['weight'] = np.where(
    sub_data_ym['label'] == 9, 1 / sub_data_ym.shape[0], -1 / sub_data_ym.shape[0]
)
print(sum(sub_data_ym['weight'] * sub_data_ym['shifted_ret']))
# %%
data = pd.read_parquet("../dat/monthly_data.parquet")
# data['P_KJX_BL'].replace("None", np.nan, inplace=True)
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data_utils import shift_prob

dat = pd.read_parquet("../dat/monthly_data.parquet").replace(["None", "nan"], np.nan)

prob_cols = [col for col in dat.columns if col.split("_")[0] == "P"]
for col in tqdm(prob_cols[:-2]):
    dat.loc[dat[col].notna(), col] = \
        dat.loc[dat[col].notna(), col].apply(
            lambda x: [
                float(val) for val in x.strip('[]').split()
            ]
        )
    dat.loc[dat[col].notna(), col] = \
        dat.loc[dat[col].notna(), col].apply(
            lambda lst: lst[1]
        )

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

dat = dat[
    [
        'PERMNO', 'date', 'SMB_HML', 'RET', 'P_KJX_SH', 'P_KJX_SM',
        'P_KJX_SL', 'P_KJX_BH', 'P_KJX_BM', 'P_KJX_BL', 'P_KJX_ALL', 'P_MC_H', 'P_MC_L'
    ]
]

nadat = dat.dropna()
for col in nadat.columns:
    if "P_KJX" in col:
        nadat.rename(
            columns={col: f"{col}(y=1)"},
            inplace=True
        )
    elif "P_MC" in col:
        nadat.rename(
            columns={col: f"P_MC(y={col.split('_')[-1]})"},
            inplace=True
        )
    else:
        pass

for col in tqdm(nadat.columns):
    if col.startswith("P_"):
        nadat[col] = shift_prob(nadat, col)

nadat[['date', 'RET', 'P_MC(y=L)']] = nadat[['date', 'RET', 'P_MC(y=L)']].astype(float)
nadat.reset_index(drop=True, inplace=True)
nadat.to_parquet("../dat/mdl_output.parquet")
# %%
import pandas as pd

label = pd.read_csv("../label/test_label_multi_10_v2.csv", index_col=0)
ff3_class = pd.read_parquet("../dat/ff3_class.parquet")
# %%
df = pd.merge(
    label, ff3_class[['fig_name', 'SMB_HML']],
    on="fig_name", how="inner"
)
df.drop_duplicates(subset=["fig_name"], inplace=True)
# %%
df.to_csv("../label/test_label_multi_10_v3.csv")
# %%
