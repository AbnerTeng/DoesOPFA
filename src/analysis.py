# %%
"""
Probability analysis
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# %%
data = pd.read_parquet("../dat/monthly_data.parquet")
data.replace("None", np.nan, inplace=True)
# %%
sub_data = data[data["P_KJX_SH"].notna()]
# %%
