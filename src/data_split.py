# %%
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
# %%
datpath = '/Users/abnerteng/Desktop/stock/full_data'
above_path = os.path.abspath(os.path.join(datpath, ".."))
test = pd.read_csv(datpath + '/dat10001.csv')
print(test.tail())

## how to deal with NaN value??
## after dealing with NaN value, separate with every month.
## plot 20k data every one month.
# %%
test.isnull().values.any()
test.isnull().sum().sum()
nan = test['DlyClose'].isnull()
nan = pd.DataFrame(nan)
print(nan[(nan['DlyClose'] == True) & (nan['DlyClose'].shift(-1) == False)])
# %%
test.drop(['Unnamed: 0'], axis=1, inplace=True)
price_dt = test[['PERMNO', 'DlyCalDt', 'DlyVol', 'DlyClose', 'DlyLow', 'DlyHigh', 'DlyOpen']]
price_dt.columns = ['PERMNO', 'Day', 'Volume', 'Close', 'Low', 'High', 'Open']
price_dt['MA'] = price_dt['Close'].rolling(window=20).mean()
price_dt[['m', 'd', 'y']] = price_dt.Day.str.split("/", expand = True)
price_dt.to_csv(above_path + '/recolumn_data/price_dt_10001.csv', index = False)
# %%[markdown]
## Do not edit codes above
# %%
df = pd.read_csv(above_path + '/recolumn_data/price_dt_10001.csv')

# %%
##df[['m', 'd', 'y']] = df.Day.str.split("/", expand = True)
# %%
def save_csv(df):
    df = df[-20: ]
    df.to_csv(above_path + '/' + str(df['PERMNO'].iloc[0]) + '/' + df['y'].iloc[0] + df['m'].iloc[0] + '.csv', index = False)
df.groupby(['y', 'm']).apply(save_csv)
# %%
##def split_dataframe(df, chunk_size = 20):
##    chunks = list()
##    num_chunks = len(df) // chunk_size + 1
##    for i in range(num_chunks):
##        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
##    return chunks
### %%
##df_split = split_dataframe(full_data, chunk_size = 20)
# %%
