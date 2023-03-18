import pandas as pd
from datetime import datetime
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

#Duration(calculate duration)
def duration_total(df_name,start_timecol,end_timecol):
  duration = []
  start_time = df_name[[start_timecol]]
  end_time = df_name[[end_timecol]]

  for i in df_name.index:

    start_hour, start_minute, start_period = map(str.strip, start_time[i].split(' '))
    end_hour, end_minute, end_period = map(str.strip, end_time[i].split(' '))
    
    start_hour = int(start_hour)
    start_minute = int(start_minute)
    end_hour = int(end_hour)
    end_minute = int(end_minute)
    
    # Convert hours to 24-hour format if necessary
    if start_period == 'PM' and start_hour != 12:
        start_hour += 12
    if end_period == 'PM' and end_hour != 12:
        end_hour += 12
    if start_period == 'AM' and start_hour == 12:
        start_hour = 0
    if end_period == 'AM' and end_hour == 12:
        end_hour = 0
    
    # Calculate time difference in minutes
    start_time_minutes = start_hour * 60 + start_minute
    end_time_minutes = end_hour * 60 + end_minute
    time_diff = end_time_minutes - start_time_minutes

    duration += [time_diff]
    
  df_name["duration"] = duration

#multiple Linear regression 
def lin_regress(df_name,df_col,waste_col, building):
  X = []
  y = []
  for i in df_name.index:
    if df_name[["Street Address"][i]] == building:
      X += [df_name[df_col][i]]
      y += [df_name[waste_col][i]]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)
  reg = LinearRegression().fit(X_train,y_train)
  print(reg.score())
  y_pred=reg.predict(X_test)
  reg.predict(np.array([["TON"]])) #Write code to find it when its a ton

  #Decision tree regressor

def tree_regress(df_name,df_col,waste_col, building):
  X = []
  y = []
  for i in df_name.index:
    if df_name[["Street Address"][i]] == building:
      X += [df_name[df_col][i]]
      y += [df_name[waste_col][i]]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)
  treg =  DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
  treg.score()
  y_pred=treg.predict(X_test) #code to find y value of 1000 based on some x (2222?) 