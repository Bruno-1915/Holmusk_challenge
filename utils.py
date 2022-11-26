import numpy as np
from numpy import nan
import pandas as pd

from constants import Diabetes_level, GFR_level

def fill_dates(df):
    t = pd.date_range(start='2000-01-01', freq='1D', periods=df.time.iloc[-1]+1).to_series()
    df['dates'] = t.iloc[list(df.time.values)].values
    t = pd.date_range(start='2000-01-01', freq='1D', periods=df.time.iloc[-1]+1).to_series()
    l = [{column: (nan if column != 'dates' else t[i]) for column in df.columns}
        for i in t]
    pp = pd.DataFrame.from_records(l)
    pp = pp.loc[[i not in df.dates.to_list() for i in pp.dates]]
    return pd.concat([pp, df]).sort_values(by='dates')

def classify_diabetes(df):
    index = df.columns.to_list().index('value_glucose')
    # [Diabetes.normal if i < 7.8 else Diabetes.prediabetes if i < 11.1 else Diabetes.diabetes]
    df.insert(loc=index, column = 'Diabetes', 
            value = [Diabetes_level.normal if i < 7.8 else Diabetes_level.prediabetes if i < 11.1 else Diabetes_level.diabetes
                    for i in df['value_glucose']])
    df = df.iloc[:,:index].join(pd.get_dummies(df.Diabetes, prefix='Diab')).join(df.iloc[:,index:])
    df.drop(columns=['Diabetes'], inplace=True)
    return df

def classify_gfr(df):
    index = df.columns.to_list().index('GFR')
    df.insert(loc=index, column = 'GFR_level', 
            value = [GFR_level.g5 if i < 16 
                    else GFR_level.g4 if i < 30 
                    else GFR_level.g3b if i < 45
                    else GFR_level.g3a if i < 60
                    else GFR_level.g2 if i < 90
                    else GFR_level.g1
                    for i in df['GFR']])
    df = df.iloc[:,:index].join(pd.get_dummies(df.GFR_level, prefix='GFR_level')).join(df.iloc[:,index:])
    df.drop(columns=['GFR_level'], inplace=True)
    return df

def get_gfr(creatinine, race, gender, age,):
    k = .7 if gender == 'Female' else .9
    alpha = -.329 if gender == 'Female' else -.411
    female = 1.018 if gender == 'Female' else 1
    race = 1.159 if race == 'Black' else 1
    return 141 * (np.minimum(creatinine / k, 1) ** alpha) * (np.maximum(creatinine / k, 1) ** -1.209) * (.993 ** age) * female * race