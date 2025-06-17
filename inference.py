import pandas as pd
import datetime,csv
from tkinter import filedialog
from churn_degree import *

# sklearn modules for data preprocessing-------------------------------------
from sklearn.preprocessing import StandardScaler 
import joblib

class inference():
    def __init__(self,database=''):
        database=pd.read_csv('dataset.csv')
        database = database.iloc[:-3652]
        self.database=database

    def getdata(self):
        print('How do you want to input:')
        ### Manual input---------------------------------------------
        # value = input("1: Choose a list of id from file (*.log, *.txt, *.csv); anything else for inference all in Database:\n")
        value=2

        id_list = []
        if value == 1:
            filename = filedialog.askopenfile(title='Please choose id list file')
            filename= filename.name
            # filename='1idlist.txt'
            with open(filename,'r',newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        id_list+=row  
        
            id_list=list(dict.fromkeys(id_list))
            id_not_in_DB = set(id_list)-set(self.database['uid'])
            if len(id_not_in_DB)>0:
                print('id not in DB:',len(id_not_in_DB),', you may want to updata your Database')
            id_list = list(set(id_list) & set(self.database['uid']))
            data={}
            data = pd.DataFrame(data)
            
            for uid in id_list:    
                # print(self.database.loc[self.database['uid'] == uid])
                data=pd.concat([data,self.database.loc[self.database['uid'] == uid]],axis=0)
            
            return data
        else:
            print('Run predict for all id in Database')
            return self.database
    
    def run_inference(self):
        
        ### load data------------------------------------------
        data = self.getdata()
        data.head()
        data.columns
        data.describe()
        data.dtypes
        data.columns.to_series().groupby(data.dtypes).groups
        data.info()
        data.isna().any()
        data["isChurn"].value_counts()

        ### Preprocess-------------------------------------------
        response = data["isChurn"]

        last_play = data['last_time_play']
        data=data.drop(['last_time_play',"isChurn"],axis=1)


        identity = data['uid']
        data=data.drop(['uid'],axis=1)
        # Step 14: Feature Scaling-----------------------------------------------------------------------
        print(data.columns)
        sc_X = StandardScaler()
        data2 = pd.DataFrame(sc_X.fit_transform(data))
        data2.columns = data.columns.values
        data2.index = data.index.values
        # data = data2

        classifier = joblib.load('finalized_model.sav')
        pred = classifier.predict(data2)
        print('------------------------------')
        # results = pd.concat([identity,data,response], axis = 1).dropna()
        final_result=pd.concat([identity,data['time_to_play']],axis=1)

        sum_time = data["length_of_sessions"] + data["interval_between_sessions"]
        final_result['active_time']=sum_time

        final_result=pd.concat([identity,data['time_to_play'],final_result['active_time'],data['high_level']],axis=1)

        final_result['predictions'] = pred 
        p=[]
        for i in last_play:
            p.append(datetime.datetime.now() - datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S"))
        now={
            'now':p 
        }
        now=pd.DataFrame(now)

        final_result['days_to_churn'] = now

        final_result=final_result[final_result['predictions']==True]

        warning = churn_degree(list(final_result['uid']))

        final_result['warning'] = list(warning.values())



        # final_results.loc[final_results['predictions'].isin('True')]
        # print(final_results['predictions'].value_counts())
        final_result.to_excel('final_result.xlsx')
        final_result.to_csv('final_result.csv')
        print('Saved to final_result.csv')





