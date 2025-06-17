import tkinter
from tkinter import filedialog
from tkinter.tix import Tk
from preprocess import *
import pandas as pd
import csv,re
from churn_cal import *
from io import StringIO
from churn_degree import *

# dataset = get_dataset(datalog='data0.log', set_parse='7', set_append=[6,7], append=0, churn_date='2022-09-20 17:00:52')
# dataset.run()
# total_time(uid = 'LrpzcXTnzuaU6wwrzhGurZyUXtA2', data_log='data0.log')
# time=[]

# df=pd.read_csv('session6.csv')
# pd.read_csv(StringIO(df.to_csv()), index_col=[0])
# df = df.drop(['Unnamed: 0'], axis=1)
# df.to_csv('session6.csv', index=False)

# # uid,level,time_to_play,ball_popped,ball_drop,shot,is_win,is_first_time_win,powerball,version_name
# for i in range(len(data['uid'])):
#     time.append('2022-09-16 23:11:28')

# data['time_get_info']=time
# data.to_csv('level6.csv', index='False')

    # uid,level,time_to_play,ball_popped,ball_drop,shot,is_win,is_first_time_win,powerball

# uids=pd.read_csv('final_result.csv')
# uids=list(uids['uid'])
# print(len(uids))
# churn_degree(uids)

set_append=[6,7]
for x in set_append:
    level_filename = 'level' + str(x) + '.csv'
    dat=pd.read_csv(level_filename)
    dict=dat['high_level'].to_dict
    print(dict)
    # high_level_info(high_level=dict,file=level_filename)

