# %%
import pandas as pd
import numpy as np
import os
import sys

# %%
datpath = '/Users/abnerteng/Desktop/stock/full_data'
above_path = os.path.abspath(os.path.join(datpath, ".."))
test = pd.read_csv(datpath + '/dat10032.csv')
print(test.head())
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
price_dt = test[['DlyVol', 'DlyClose', 'DlyLow', 'DlyHigh', 'DlyOpen']]
price_dt.columns = ['Volume', 'Close', 'Low', 'High', 'Open']
##price_dt.drop([6058, 6059], axis=0, inplace=True)
# %%
price_dt['MA'] = price_dt['Close'].rolling(window=20).mean()
# %%
price_dt.to_csv(above_path + '/recolumn_data/price_dt_10032.csv', index = False)
# %%
df = pd.read_csv(above_path + '/recolumn_data/price_dt_10032.csv')
# %%
df.columns = ['Volume', 'Close', 'Low', 'High', 'Open', 'MA']
full_data = df[999:]

# %%
def split_dataframe(df, chunk_size = 20):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks
# %%
df_split = split_dataframe(full_data, chunk_size = 20)

# %%
for i in range(290):
    df_split[i].to_csv(above_path + '/split_data_10032/10032_split' + str(i) + '.csv', index = False)
# %%
