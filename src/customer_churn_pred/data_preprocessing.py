import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OrdinalEncoder

def load_data(url):
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    y = df['Churn']
    X = df.drop(['customerID', 'Churn'], axis=1)
    
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)
    
    oe = OrdinalEncoder()
    X = oe.fit_transform(X)
    
    return X, y