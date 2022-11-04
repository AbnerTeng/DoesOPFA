import os
import pandas as pd

path = os.getcwd()
data_path = '/Users/abnerteng/Downloads'
abspath = os.path.abspath(os.path.join(data_path, os.pardir))

df = pd.read_csv(data_path + '/WRDS_data.csv')

name = df.groupby('PERMNO')
size = name.size()
size = pd.DataFrame(size)
size['PERMNO'] = size.index

for i in size['PERMNO']:
    locals()['dat' + str(i)] = name.get_group(i)

for i in size['PERMNO']:
    locals()['dat' + str(i)].to_csv(data_path + '/dat' + str(i) + '.csv')