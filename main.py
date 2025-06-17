from preprocess import *
# from inference import *
from inference import *
from churn_degree import *
### Make trainning dataset.csv------------------------------------------
# datalog = data log file
# set_parse=1: write paresed data to level1.csv, session1.csv, uid_play1.csv
# set_append=[0,1,2]: process data level0, level1, level2, session..., uid_play... to level.csv, session.csv, uid_level.csv
# append = 0: remove old level.csv, session.csv, uid_play.csv, else combine.
# churn_date: the latest day data collected to calculate churn
# churn_date = '2022-09-15 09:49:02' # for dataset0
data = get_dataset(datalog='data0.log', set_parse='7', set_append=[6,7], append=0, churn_date='2022-09-23 10:00:00')
data.process_data()

### Trainning
#run in ipynb


### Inference
# getresult = inference()
# getresult.run_inference()


###
# print(churn_degree('050esJCBNDTmqKbEP0FI4tfkAvy1'))
# adjust_from_dataset()
