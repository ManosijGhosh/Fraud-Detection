'''
AUTHOR: SOUMYADEEP THAKUR
DATE: 6 OCT 2018
'''

import os
import numpy as np
import signal
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from confusion import evaluate
from knnClassifier import createKnnModel
from autoencoder_kfold import autoencoder

'''
TO DO: DROP TIME ATTRIBUTE AND TRY
'''

def preprocess():
	
	df = pd.read_csv('creditcard.csv')
	data = df
	data['Time'] = data['Time'].apply(lambda x : ((x//60)%1440)/1440) # convert to time of day in minuites
	data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
	
	# Seperate X and Y data

	Y_val = data['Class']
	X_val = data.drop(['Class'], axis=1)
	X_val = X_val.values
	Y_val = Y_val.values

	return X_val, Y_val

def split(X_val, Y_val,fold):

    # Prepare for stratified KFold crossvalidation

	skf = StratifiedKFold(n_splits=fold)
	skf.get_n_splits(X_val, Y_val)
	#print(skf)

	for train_index, test_index in skf.split(X_val, Y_val):
		yield (train_index, test_index)

def consfusion_eval(actualLabels,file,knnLabels):

	#sprint(labels)
	#TPX, TNX, FPX, FNX = list(), list(), list(), list()
	
	for i in range(0,20):
		finalLables = list()
		labels_encoder = evaluate(file, (i/10))
		print(len(actualLabels),' ',len(labels_encoder),' ',len(knnLabels))
		if (len(labels_encoder)!=len(knnLabels)):
			#print ('error two classifiers have different label output')
		for j in range(len(labels_encoder)):
			if (knnLabels[j]==0):
				finalLables.append(0)
			else:
				finalLables.append(labels_encoder[j])
		if (len(actualLabels)!=len(finalLables)):
			#print ('error final label not same as actual')
		tp = tn = fp = fn = 0
		for j in range(len(finalLables)):
			if (finalLables[j]==actualLabels[j]):
				if finalLables[j]==0:
					tp+=1
				else:
					tn+=1
			else:
				if actualLabels[j]==1:
					fp+=1
				else:
					fn+=1;
		print('Thresh: ', (i/10), 'Conf: ', tp, ' -- ', tn, ' -- ', fp, ' -- ', fn)
		
def main():
	
	#X_test, X_train, Y_test = make_data()
	if not os.path.exists('models'):
		os.mkdir('models')
	X_val, Y_val = preprocess()
	fold = 2
	count=0
	str = 'test_log_mano_50'
	for train_index, test_index in split(X_val, Y_val, fold):
		count += 1
		X_train, X_test = X_val[train_index], X_val[test_index]
		Y_train, Y_test = Y_val[train_index], Y_val[test_index]

		print('splitting done')
		knnLabels = createKnnModel(X_train, X_test, Y_train)
		knnLabels = [1] * len(Y_test)

		print('knn tested')
		autoencoder(X_train, X_test, Y_train, Y_test, str, count)
		consfusion_eval(Y_test, ('models/'+str+'_%i') %count,knnLabels)
		#'''
		

main()

## THRESH 1.0: true fraud: 434, false fraud: 21397, total: 284800