import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def feature_engineering_f4(df):
    df['Arrival_Hour'] = df['x_ArrivalDTTM'].dt.hour
    df['Arrival_DayOfWeek'] = df['x_ArrivalDTTM'].dt.dayofweek
    df['Scheduled_Arrival_Duration'] = (df['x_ArrivalDTTM'] - df['x_ScheduledDTTM']).dt.total_seconds()
    df['SumCountScanned'] = (df['LineCount0'] + df['LineCount1'] + df['LineCount2'] + df['LineCount3'] + df['LineCount4']) * df['NumScannersUsedToday']
    df['SumScannersUsed'] = df['SumWaits'] / df['NumScannersUsedToday']
    
    df['Total_LineCount'] = df[['LineCount0', 'LineCount1', 'LineCount2', 'LineCount3', 'LineCount4']].sum(axis=1)
    df['Avg_FlowCount'] = df[['FlowCount2', 'FlowCount4']].mean(axis=1)
    df['Wait_Multiplied_AvgWait'] = df['Wait'] * df['AvgWaitLastK1Customers']
    
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    scaler = RobustScaler()
    numeric_features = ['Total_LineCount', 'Avg_FlowCount', 'Wait_Multiplied_AvgWait']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df