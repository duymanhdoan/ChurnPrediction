import datetime
import re,csv

# Decide if uid has churned
def churn(time_last_play,date):
    if date == '':
        time = datetime.datetime.now()
    else: time = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    id_churn=[]
    for x in time_last_play:
        last_play = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        if (time - last_play) > datetime.timedelta(days = 14):
            id_churn.append(True)
        else: id_churn.append(False)
    return id_churn

### Total ball pop,drop in a level
def total_ball(str):
    balls = re.split("\,", str[1:len(str)-1])
    totalball=0
    for ball in balls:
        totalball += float(ball)
    return totalball

### Save highest level info
def high_level_info(high_level,file):
    with open(file,'r',newline='') as f:
        # uid,level,time_to_play,ball_popped,ball_drop,shot,is_win,is_first_time_win,powerball
        reader = csv.reader(f)
        next(f)
        for row in reader:
            pass