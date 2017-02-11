#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
features = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=111)
crossvalidation = KFold(n=X_train.shape[0], n_folds=5,shuffle=True, random_state=1)

maxSize = 0;
maxScore = 0.0;

for size in range(1,10):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(size,), random_state=1)
	
	score = np.mean(cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
	if(score > maxScore):
			maxScore = score
			maxSize = size	
	print 'hidden_layer_sizes:{0}  Accuracy: {1:.3f}'.format(size,score) #to identify the best hidden_layer_sizes iteratively

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(maxSize,), random_state=1)

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

## Checking the accuracy
print(accuracy_score(predicted, y_test))