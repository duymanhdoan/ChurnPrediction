from math import floor
import pandas as pd
import csv

### Return 2 dictionary --------------------------------------------------
# dict[uid]=[time,ball,shot,win] 
# high_level[uid]=highest level played
def uid_value(uids):
    dataset={}
    with open('dataset.csv','r',newline='') as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            if row[0] in uids:
                dataset[row[0]]=[float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11]),float(row[12]),float(row[13]),row[14],row[15]]
    
    level={}
    high_level={}
    with open('level.csv','r',newline='') as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            if (row[0] in uids):
                level[row[0]]=float(row[10])
                high_level[row[0]]=float(row[1])

    with open('level0.csv.bk','r',newline='') as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            if (row[0] in uids):
                if (row[0] not in level.keys()): level[row[0]]=1
                else: level.update({row[0]: level[row[0]]+1})

    

    dict={}
    for uid in uids:
        if dataset[uid][2]==0: 
            time=0
            ball=0
            shot=0
            win=0
        else:
            
            time = dataset[uid][2]/(dataset[uid][10]+dataset[uid][11])
            
            ### Data error here: ball pop, drop >0 but shot = 0. Assum 20 shot per play
            if dataset[uid][5]==0: ball = (dataset[uid][3]+dataset[uid][4]*1.25)/(level[uid]*20)
            else: ball = (dataset[uid][3]+dataset[uid][4]*1.25)/dataset[uid][5]

            shot = dataset[uid][5]/level[uid]
            
            win = dataset[uid][6]/level[uid] + (dataset[uid][7])*1.5/level[uid]
        dict[uid]=[time,ball,shot,win]
    
    # dict[uid]: [time,ball,shot,win]
    return dict,high_level

### Calculate average values of active players-----------------------------
def adjust_from_dataset():
    avg_time=0
    avg_ball=0
    avg_shot=0
    avg_win=0
    
    dataset = pd.read_csv('dataset.csv')
    dataset=dataset[dataset['isChurn']==False]
    uids = list(dataset['uid'])
    get_values, dummy = uid_value(uids)
    for uid in uids:
        avg_time=avg_time+get_values[uid][0]
        avg_ball=avg_ball+get_values[uid][1]
        avg_shot=avg_shot+get_values[uid][2]
        avg_win=avg_win+get_values[uid][3]
    
    avg_time=avg_time/len(uids)
    avg_ball=avg_ball/len(uids)
    avg_shot=avg_shot/len(uids)
    avg_win=avg_win/len(uids)
    return [avg_time,avg_ball,avg_shot,avg_win]

### Churn degree and warning----------------------------------------------------------
def churn_degree(uids):
    # print('calculate average')
    avg = adjust_from_dataset()
    # avg=[0.13832505556157368, 118.35220205363143, 96.44966186726977, 6.759424382561003]
    # print('finish, average = ', avg)
    id_values, high_level = uid_value(uids)
    
    dict={}
    for uid in uids:
        result=''
        
        time_perct=(id_values[uid][0]/avg[0])*100
        if time_perct==0: 
            result='player does not play!'
        else:
            ball_perct=(id_values[uid][1]/avg[1])*100
            shot_perct=(id_values[uid][2]/avg[2])*100
            win_perct=(id_values[uid][3]/avg[3])*100
            # print('ball',ball_perct)
            
            result='this player '
            if time_perct<50: result=result+('play time is only {}% of average, ').format(round(time_perct,2))

            skill=ball_perct
            if high_level[uid]<11: skill=skill*1/(12-float(high_level[uid]))
            if (skill<50)or(skill>175): result = result + ('skill is {}% of average, ').format(round(skill,2))
            
            if (win_perct<50) or (win_perct>200): result=result+('win {}% of average, ').format(round(win_perct,2))

            win=0
            if win_perct<100: 
                if win_perct==0:
                    win_perct=1
                win=(100-win_perct)*(100/win_perct)
            else:
                win=win_perct-100
                if win_perct==100: win=1

            ball=0
            if ball_perct<100: 
                if ball_perct==0:
                    # print(win_perct,uid)
                    ball_perct=1
                ball=(100-ball_perct)*(100/ball_perct)
            else:
                if ball_perct==150: 
                    ball=1
                else: ball=abs(150-ball_perct)
            
            result=result+('churn level is {}').format(round((1/time_perct)*ball*win,2))

            # print(result)

        dict[uid]=result

    return dict

