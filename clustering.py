from sklearn.cluster import DBSCAN
import numpy as np
import pandas
from sklearn.model_selection import train_test_split

def normalize(df):
    result = df.copy()
    (a,b)= np.shape(df)
    for sampleIndex in range(a):
    	feature[sampleIndex,1] = (feature[sampleIndex,1]/(60*60))%24

    for featureIndex in range(b):
        max_value = df[:,featureIndex].max()
        min_value = df[:,featureIndex].min()
        result[:,featureIndex] = (df[:,featureIndex] - min_value) / (max_value - min_value)
    return result

fileName='creditcard.csv'
dataframe=pandas.read_csv(fileName)
(a,b)= np.shape(dataframe)
print (a, b)
#print(dataframe)

feature = dataframe.values[:,0:b-1]
feature = normalize(feature)
#print(feature)
target = dataframe.values[:,b-1]

fold = 3

X_train, X_test, y_train, y_test = train_test_split(feature, target,stratify=target ,test_size=1.0/fold)

clustering = DBSCAN(eps=3, min_samples=2).fit(X_train)
print(clustering)