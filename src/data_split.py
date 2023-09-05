import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from os import listdir

datpath = '/Users/abnerteng/Desktop/CNN/dat/WRDS_split_data'
abvpath = '/Users/abnerteng/Desktop/CNN/dat'
recolumnpath = '/Users/abnerteng/Desktop/CNN/dat/recolumn_data'
list_ = listdir(datpath)
"""
how to deal with NaN value?? ==> ignore
after dealing with NaN value, separate with every month.
plot 20k data every one month.
"""
## test.isnull().values.any()
## test.isnull().sum()
## nan = test['DlyClose'].isnull()
## nan = pd.DataFrame(nan)
## print(nan[(nan['DlyClose'] == True) & (nan['DlyClose'].shift(-1) == False)])

class recolumn():
    def __init__(self, datpath, list_, abvpath, recolumnpath):
        self.datpath = datpath
        self.list_ = list_
        self.abvpath = abvpath
        self.recolumnpath = recolumnpath

    def process(self):
        for i in range(len(self.list_)):
            df = pd.read_csv(self.datpath + '/' + self.list_[i])
            price_dt = df[['PERMNO', 'DlyCalDt', 'DlyVol', 'DlyClose', 'DlyLow', 'DlyHigh', 'DlyOpen']]
            price_dt.columns = ['ID', 'Day', 'Volume', 'Close', 'Low', 'High', 'Open']
            price_dt['MA'] = price_dt['Close'].rolling(window=20).mean()
            price_dt['Day'] = pd.to_datetime(price_dt['Day'], format = '%m/%d/%Y')
            price_dt['y'] = price_dt['Day'].dt.year
            price_dt['m'] = price_dt['Day'].dt.month
            price_dt['d'] = price_dt['Day'].dt.day
            price_dt.to_csv(self.recolumnpath + '/' + self.list_[i], index = False)

recolumn(datpath, list_, abvpath, recolumnpath).process()
"""
Do not edit codes above
"""

## df = pd.read_csv(datpath + '/recolumn_data/AAPL.csv')

## def save_csv(df):
##     df = df[-20: ]
##     df.to_csv(datpath + '/' + str(df['Ticker'].iloc[0]) + '/' + df['y'].iloc[0] + df['m'].iloc[0] + '.csv', index = False)
## df.groupby(['y', 'm']).apply(save_csv)

## def save_csv(df):
##     if df['m'].iloc[0] >= 10:
##         df.to_csv(datpath + '/' + str(df['Ticker'].iloc[0]) + '/' + str(df['y'].iloc[0]) + str(df['m'].iloc[0]) + '.csv', index = False)
##     else:
##         df.to_csv(datpath + '/' + str(df['Ticker'].iloc[0]) + '/' + str(df['y'].iloc[0]) + '0' + str(df['m'].iloc[0]) + '.csv', index = False)
## df.groupby(['y', 'm']).apply(save_csv)

## def split_dataframe(df, chunk_size = 20):
##    chunks = list()
##    num_chunks = len(df) // chunk_size + 1
##    for i in range(num_chunks):
##        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
##    return chunks

## df_split = split_dataframe(full_data, chunk_size = 20)
