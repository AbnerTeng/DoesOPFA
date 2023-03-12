# %%
import os
## import polars as pl
import pandas as pd 

path = os.getcwd()
data_path = '/Users/abnerteng/Desktop/CNN/dat/WRDS_origin_dat.csv'
split_data_path = '/Users/abnerteng/Desktop/CNN/dat/WRDS_split_data'

df = pd.read_csv(data_path)
# %%
class split_data:
    def __init__(self, df, split_data_path):
        self.df = df
        self.split_data_path = split_data_path
    
    def split(self):
        name = self.df.groupby('PERMNO')
        size = name.size()
        size = pd.DataFrame(size)
        size['PERMNO'] = size.index

        for i in size['PERMNO']:
            locals()[str(i)] = name.get_group(i)
            locals()[str(i)].to_csv(self.split_data_path + '/' + str(i) + '.csv', index = False)
# %%
action = split_data(df, split_data_path)
action.split()
# %%
