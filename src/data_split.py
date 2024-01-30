import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from os import listdir
from tqdm import tqdm

## originpath = '/Users/abnerteng/Desktop/CNN/dat/WRDS_origin_dat.csv'
datpath = '/Users/abnerteng/Desktop/CNN/dat/WRDS_split_data'
abvpath = '/Users/abnerteng/Desktop/CNN/dat'
recolumnpath = '/Users/abnerteng/Desktop/CNN/dat/recolumn_data'
list_ = listdir(datpath)
relist = listdir(recolumnpath)
for_testpath = '/Users/abnerteng/Desktop/CNN/dat/for_test'
## segpath = '/Users/abnerteng/Desktop/CNN/dat/segment_dat'
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

class preprocess():
    def __init__(self):
        ## self.originpath = originpath
        self.datpath = datpath
        self.list_ = list_
        self.abvpath = abvpath
        self.recolumnpath = recolumnpath
        self.for_testpath = for_testpath
        ## self.orgdf = pd.read_csv(self.originpath)
        self.relist = relist

    def split(self):
        name = self.orgdf.groupby('PERMNO')
        size = pd.DataFrame(name.size())
        size['PERMNO'] = size.index

        for i in size['PERMNO']:
            locals()[str(i)] = name.get_group(i)
            locals()[str(i)].to_csv(self.datpath + '/' + str(i) + '.csv', index = False)

    def recolumn(self):
        try:
            for i in tqdm.trange(len(self.list_)):
                df = pd.read_csv(self.datpath + '/' + self.list_[i])
                df.dropna(inplace = True)
                price_dt = df[['PERMNO', 'DlyCalDt', 'DlyVol', 'DlyClose', 'DlyLow', 'DlyHigh', 'DlyOpen']]
                price_dt.columns = ['ID', 'Day', 'Volume', 'Close', 'Low', 'High', 'Open']
                price_dt['MA'] = price_dt['Close'].rolling(window=20).mean()
                price_dt['Day'] = pd.to_datetime(price_dt['Day'], format = '%m/%d/%Y')
                price_dt['y'] = price_dt['Day'].dt.year
                price_dt['m'] = price_dt['Day'].dt.month
                price_dt['d'] = price_dt['Day'].dt.day
                price_dt.to_csv(self.recolumnpath + '/' + self.list_[i], index = False)
                print(f'{self.list_[i]} process complete')
        except:
            print(f'{self.list_[i]} process error')

    def for_test(self):
        for i in tqdm(self.relist):
            df = pd.read_csv(self.recolumnpath + '/' + i)
            try:
                df = df[df['y'] > 2000]
                df['ym'] = df['y'] * 100 + df['m']
                ym = df['ym'].unique()
                group = df.groupby('ym')

                if not os.path.exists(self.for_testpath + '/' + str(i)[:5]):
                        os.makedirs(self.for_testpath + '/' + str(i)[:5])

                for j in ym:
                    ## make folders
                        group.get_group(j).to_csv(self.for_testpath + '/' + str(i)[:5] + '/' + str(j) + '.csv', index = False)
            except:
                print(f'{self.relist[i]} No data after 2000')


            


if __name__ == "__main__":
    act = preprocess()
    act.for_test()