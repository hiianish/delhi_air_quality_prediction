import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("AQI DATASET.csv")

df["Date_time"] = pd.to_datetime(dict(year=df["Year"],month= df["Month"],day = df["Date"]))
df = df.sort_values('Date_time').reset_index(drop = True)

df["Day_of_week"] = df["Date_time"].dt.dayofweek
df["is_weekend"] = df["Day_of_week"].isin([5,6]).astype(int)

def get_season(Month):
    if Month in [12,1,2]:
        return "Winter"
    elif Month in[3,4,5]:
        return "Summer"
    elif Month in [6,7,8]:
        return "Monsoon"
    else:
        return "Post_monsoon"
    
df["Season"] = df["Month"].apply(get_season)
df['is_holiday'] = (df['Holidays_Count'] > 0).astype(int)
df.to_csv("clean_dataset1",index = False)


df1 = pd.read_csv("clean_dataset1")
df1.drop(columns=['Date', 'Days', 'Holidays_Count'], inplace=True)
print(df1)

pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']

for col in pollutants:
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df1[col] = np.clip(df[col], lower, upper)


for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']:
    df1[f'{col}_roll3'] = df[col].rolling(window=3, min_periods=1).mean()


for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']:
    df1[f'{col}_lag1'] = df1[col].shift(1)

df1.dropna(inplace=True)
print(df1)

df1.to_csv("final_dataset",index = False)
dataset = pd.read_csv("final_dataset")
dataset