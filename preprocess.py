from asyncore import write
from operator import concat
import pandas as pd
import csv,re,os,datetime
from genericpath import exists
from churn_cal import *

# Make trainning dataset.csv
# set_parse=1: write data to level1.csv, session1.csv, uid_play1.csv
# set_append=[0,1,2]: process data level0, level1, level2, session..., uid_play... to level.csv 
# append = 0: remove old level.csv, session.csv, uid_play.csv, else combine.
# churn_date: the latest day data collected to calculate churn
# churn_date = '2022-09-15 09:49:02' # for dataset0
class get_dataset():
    def __init__(self, datalog,set_parse='x',set_append=[''],append=0,churn_date='2022-09-20 17:00:52'):
        self.datalog = datalog
        self.set_parse = set_parse
        self.set_append = set_append
        self.append = append
        self.churn_date = churn_date

    # parse data from log------------------------------------------
    # set=1: write data to level1.csv, session1.csv, uid_play1.csv
    def parse_data(self):
        level_filename = 'level' + str(self.set_parse) + '.csv'
        session_filename = 'session' + str(self.set_parse) + '.csv'
        # uid_play_filename = 'uid_play' + str(self.set_parse) + '.csv'
        print('Parse data to files: ',level_filename,session_filename)

        with open(level_filename,'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['uid', 'level', 'time_to_play', 'ball_popped', 'ball_drop', 'shot', 'is_win', 'is_first_time_win', 'powerball', 'time_get_info'])
        with open(session_filename,'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['uid', 'number_of_sessions', 'length_of_sessions', 'interval_between_sessions', 'playCount','time_get_info'])
        # with open(uid_play_filename,'w', newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['uid', 'play_count'])

        playCount = {}
        file = open(self.datalog, 'r')
        lines = file.readlines()
        for row in lines:

            # Parse level----------------------------------------------
            if row.find('SaveLevel') > -1:
                uid=row[row.find('uid=')+4:row.find('&msg')]
                s=row[row.find('level'):]
                level=s[9:s.find(',')]
                s=row[row.find('time_to_play'):]
                time_to_play=s[16:s.find(',')]
                s=row[row.find('ball_popped'):]
                ball_popped=s[15:s.find(',%')]
                s=row[row.find('ball_drop'):]
                ball_drop=s[13:s.find(',%')]
                s=row[row.find('shot'):]
                shot=s[9:s.find(',')]
                s=row[row.find('is_win'):]
                is_win=s[10:s.find(',')]
                s=row[row.find('is_first_time_win'):]
                is_first_time_win=s[21:s.find(',')]
                s=row[row.find('powerball'):]
                if s.find(',%22date_login') >-1:
                    powerball=s[13:s.find(',%22date_login')]
                else:
                    powerball=s[13:s.find('%7D')]
                time = row[10:29]

                data = [uid, level, time_to_play, ball_popped, ball_drop, shot, is_win, is_first_time_win, powerball,time]
                with open(level_filename,'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)

            # Parse Session----------------------------------------------
            if row.find('StartSession') > -1:
                uid=row[row.find('uid=')+4:row.find('&msg')]
                s=row[row.find('number_of_sessions'):]
                number_of_sessions=s[22:s.find(',')]
                s=row[row.find('length_of_sessions'):]
                length_of_sessions=s[22:s.find(',')]
                s=row[row.find('interval_between_sessions'):]
                interval_between_sessions=s[29:s.find(',')]
                s=row[row.find('playCount'):]
                playCount=s[13:s.find(',')]
                # s=row[row.find('levelEnd'):]
                # levelEnd=s[12:s.find(',')]
                # s=row[row.find('version_name'):]
                # version_name = s[19:s.find('%22,%22p')]
                time = row[10:29]

                data = [uid, number_of_sessions, length_of_sessions, interval_between_sessions,playCount,time]
                with open(session_filename,'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                # with open(uid_play_filename,'a', newline="") as f:
                #     writer = csv.writer(f)
                #     writer.writerow([uid,time])   

    # process data ------------------------------------------
    # set=[0,1,2]: process data level0, level1, level2, session..., uid_play... to level.csv 
    # append = 0: remove old level.csv, session.csv, uid_play.csv, else combine.
    def process_data(self):
        for x in self.set_append:
            level_filename = 'level' + str(x) + '.csv'
            session_filename = 'session' + str(x) + '.csv'
            # uid_play_filename = 'uid_play' + str(x) + '.csv'
            if self.append==0: print('Removed file level.csv, session.csv, uid_play.csv. Processing data...')
            else: print('Processing data, combine with level.csv, session.csv, uid_play.csv...')

            # combine levelx.csv data to level.csv
            dict_level = {}
            if exists('level.csv'):
                if self.append == 0:
                    os.remove("level.csv")
                else:
                    # uid,level,time_to_play,ball_popped,ball_drop,shot,is_win,is_first_time_win,powerball
                    with open('level.csv','r',newline='') as f:
                        reader = csv.reader(f)
                        next(f)
                        for row in reader:
                            dict_level[row[0]]=[row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9]]

            with open(level_filename,'r',newline='') as f:
                # uid,level,time_to_play,ball_popped,ball_drop,shot,is_win,is_first_time_win,powerball
                    reader = csv.reader(f)
                    next(f)
                    for row in reader:
                        # canculate total ball pop
                        row[3]=total_ball(row[3])
                        # canculate total ball drop
                        row[4]=total_ball(row[4])
                        
                        if row[6] == 'true': row[6]=1
                        else: row[6]=0
                        if row[7] == 'true': row[7]=1
                        else: row[7]=0

                        if row[0] not in dict_level.keys():
                            dict_level[row[0]]=[row[1],row[1],round(float(row[2]),2),row[3],row[4],row[5],row[6],row[7],row[8],1,row[9],row[9]]
                        else:
                            # uid,level,time_to_play,ball_popped,ball_drop,shot,is_win,is_first_time_win,powerball
                            # high level
                            y=[dict_level[row[0]][0]]
                            if float(dict_level[row[0]][0]) < float(row[1]): y[0]=float(row[1])
                            # total level
                            y.append(float(row[1])+float(dict_level[row[0]][1]))

                            y.append(round(float(row[2])+float(dict_level[row[0]][2]),2))
                            y.append(float(row[3])+float(dict_level[row[0]][3]))
                            y.append(float(row[4])+float(dict_level[row[0]][4]))
                            y.append(float(row[5])+float(dict_level[row[0]][5]))
                            y.append(row[6]+float(dict_level[row[0]][6]))
                            y.append(row[7]+float(dict_level[row[0]][7]))
                            y.append(float(row[8])+float(dict_level[row[0]][8]))
                            y.append(dict_level[row[0]][9]+1)
                            y.append(dict_level[row[0]][10])
                            y.append(row[9])
                            dict_level.update({row[0]: y})
            high_level = []
            total_level = []
            time_to_play = []
            ball_popped = []
            ball_drop = []
            shot = []
            is_win = []
            is_first_time_win = []
            powerball = []
            count_lines=[]
            time_first_play=[]
            time_last_play=[]

            for z in dict_level.keys():
                high_level.append(dict_level[z][0])
                total_level.append(dict_level[z][1])
                time_to_play.append(dict_level[z][2])
                ball_popped.append(dict_level[z][3])
                ball_drop.append(dict_level[z][4])
                shot.append(dict_level[z][5])
                is_win.append(dict_level[z][6])
                is_first_time_win.append(dict_level[z][7])
                powerball.append(dict_level[z][8])
                count_lines.append(dict_level[z][9])
                time_first_play.append(dict_level[z][10])
                time_last_play.append(dict_level[z][11])
            data = {
                'uid': dict_level.keys(),
                'high_level': high_level,
                'total_level': total_level,
                'time_to_play': time_to_play,
                'ball_popped': ball_popped,
                'ball_drop': ball_drop,
                'shot': shot,
                'is_win': is_win,
                'is_first_time_win': is_first_time_win,
                'powerball': powerball,
                'count_lines': count_lines,
                'time_first_play': time_first_play,
                'time_last_play': time_last_play
            }

            update_data_level = pd.DataFrame(data)
            dict_high_level=update_data_level['high_level'].to_dict            
            high_level_info(high_level=dict_high_level,file=level_filename)

            # combine sessionx.csv to session.csv
            dict_session = {}
            if exists('session.csv'):
                if self.append == 0:
                    os.remove("session.csv")
                else:
                # uid,number_of_sessions,length_of_sessions,interval_between_sessions,playCount,last_time_play
                    with open('session.csv','r',newline='') as f:
                        reader = csv.reader(f)
                        next(f)
                        for row in reader:
                            dict_session[row[0]]=[row[1],row[2],row[3],row[4],row[5],row[6]]

            with open(session_filename,'r',newline='') as f:
                # uid,number_of_sessions,length_of_sessions,interval_between_sessions,playCount,last_time_play
                reader = csv.reader(f)
                next(f)
                for row in reader:
                    if row[0] not in dict_session.keys():
                        dict_session[row[0]]=[row[1],round(float(row[2]),2),round(float(row[3]),2),row[4],row[5],row[5]]
                    else:
                        row[1]=float(row[1])+float(dict_session[row[0]][0])
                        row[2]=round(float(row[2])+float(dict_session[row[0]][1]),2)
                        row[3]=round(float(row[3])+float(dict_session[row[0]][2]),2)
                        row[4]=float(row[4])+float(dict_session[row[0]][3])
                        first_time_play=dict_session[row[0]][5]
                        dict_session.update({row[0]: [row[1],row[2],row[3],row[4],row[5],first_time_play]})

            number_of_sessions = []
            length_of_sessions = []
            interval_between_sessions = []
            playCount = []
            last_time_play = []
            first_time_play = []

            for row in dict_session.keys():
                number_of_sessions.append(dict_session[row][0])
                length_of_sessions.append(dict_session[row][1])
                interval_between_sessions.append(dict_session[row][2])
                playCount.append(dict_session[row][3])
                last_time_play.append(dict_session[row][4])
                first_time_play.append(dict_session[row][5])

            data = {
                'uid': dict_session.keys(),
                'number_of_sessions': number_of_sessions,
                'length_of_sessions': length_of_sessions,
                'interval_between_sessions': interval_between_sessions,
                'playCount': playCount,
                'last_time_play': last_time_play,
                'first_time_play': first_time_play
            }

            update_data_session = pd.DataFrame(data)


        ### Makeup missing data-------------------------------------------
        # uid with no session data
        uid_missing_session = list(set(dict_level.keys())-set(dict_session.keys()))
        number_of_sessions = []
        length_of_sessions = []
        interval_between_sessions = []
        playCount = []
        last_time_play = []
        for uid in uid_missing_session:
            if dict_level[uid][9]==1: number_of_sessions.append(1)
            else: number_of_sessions.append(int(dict_level[uid][9]*0.75))
            length_of_sessions.append(round(float(dict_level[uid][2])*1.15,2))
            intervalbetweensessions = datetime.datetime.strptime(dict_level[uid][11], "%Y-%m-%d %H:%M:%S")-datetime.datetime.strptime(dict_level[uid][10], "%Y-%m-%d %H:%M:%S")
            intervalbetweensessions = intervalbetweensessions.total_seconds()/60 - float(dict_level[uid][2])
            if intervalbetweensessions>0: interval_between_sessions.append(round(intervalbetweensessions*0.95,2))
            else: interval_between_sessions.append(0)
            playCount.append(dict_level[uid][9])
            last_time_play.append(dict_level[uid][11])
        data = {
                'uid': uid_missing_session,
                'number_of_sessions': number_of_sessions,
                'length_of_sessions': length_of_sessions,
                'interval_between_sessions': interval_between_sessions,
                'playCount': playCount,
                'last_time_play': last_time_play
            }
        lost_session_data = pd.DataFrame(data)

        # uid with no level data
        uid_no_level_data = list(set(dict_session.keys())-set(dict_level.keys()))
        none = []
        time_first_play=[]
        time_last_play=[]

        for uid in uid_no_level_data:
            none.append(0)
            time_last_play.append(dict_session[uid][4])
            time_first_play.append(dict_session[uid][5])
            
        data = {
                'uid': uid_no_level_data,
                'high_level': none,
                'total_level': none,
                'time_to_play': none,
                'ball_popped': none,
                'ball_drop': none,
                'shot': none,
                'is_win': none,
                'is_first_time_win': none,
                'powerball': none,
                'count_lines': none,
                'time_first_play': time_first_play,
                'time_last_play': time_last_play
            }
        no_level_data = pd.DataFrame(data)

        # Saved to session.csv, level.csv---------------------------------------------
        update_data_session = pd.concat([update_data_session, lost_session_data], axis = 0)
        update_data_session = update_data_session.sort_values(by=['uid'])
        update_data_session.to_csv('session.csv', index=False, mode='w')
        update_data_level = pd.concat([update_data_level,no_level_data],axis=0)
        update_data_level = update_data_level.sort_values(by=['uid'])
        update_data_level.to_csv('level.csv', index=False, mode='w')

        print('uid in level data: ',len(dict_level.keys()))
        print('uid in session data: ',len(dict_session.keys()))
        print('uid with no session data: ',len(uid_missing_session))
        print('uid with no level data: ',len(uid_no_level_data))
        print('Saved to level.csv, session.csv')
        print(len(update_data_level['uid']))
        print(len(update_data_level['uid']))
   
        
        dict_high_level = dict(zip(update_data_level['uid'], update_data_level['high_level']))
        print(type(dict_high_level))         
        high_level_info(high_level=dict_high_level,file=level_filename)

    # Make dataset for train: dataset.csv------------------------------------------
    # churn_date: the latest day data collected to calculate churn
    # churn_date = '2022-09-15 09:49:02' # for dataset0
    def make_dataset(self):
       
        dataset_filename = 'inference.csv'
       
        print('Making dataset')
        
        level = pd.read_csv('level.csv')
        session = pd.read_csv('session.csv')

        print('Number of id in level, session:',len(level['uid']),len(session['uid']))

        isChurn = churn(time_last_play=session['last_time_play'].to_list(),date=self.churn_date)
        
        dataset={
            'isChurn': isChurn
        }
        dataset = pd.DataFrame(dataset)

        level = level.drop(['count_lines','time_first_play','time_last_play'],axis=1)
        # session =  session.drop(['uid','last_time_play','first_time_play'],axis=1)
        session =  session.drop(['uid','first_time_play'],axis=1)

        dataset = pd.concat([level,session,dataset],axis=1)
        dataset.to_csv(dataset_filename, index=False, mode='w')

        
        print('Data set saved!')

    def run(self):
        self.parse_data()
        self.process_data()
        self.make_dataset()
  