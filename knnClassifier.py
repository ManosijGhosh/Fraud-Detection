from sklearn.neighbors import KNeighborsClassifier

def createKnnModel(X_train, X_test, Y_train):
	knnModel = KNeighborsClassifier(n_neighbors=3)
	knnModel.fit(X_train, Y_train)
	print('knnTrained')
	results = knnModel.predict(X_test)
	return results