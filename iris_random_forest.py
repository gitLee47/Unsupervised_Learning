## Random forest classifier  - iris dataset
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

## Loading the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X,   y, test_size=0.20, random_state=111)
crossvalidation = KFold(n=X_train.shape[0], n_folds=5,shuffle=True, random_state=1)

maxScore = 0.0;
maxTreeSet = 0;

for treeset in np.arange(10, 110, 10):
	clf = RandomForestClassifier(n_estimators=treeset)
	
	score = np.mean(cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
	if(score > maxScore):
			maxScore = score
			maxTreeSet= treeset	
	print 'n_estimators:{0}  Accuracy: {1:.3f}'.format(treeset,score) #to identify the best depth iteratively


## Building the model
clf = RandomForestClassifier(n_estimators=maxTreeSet)

## Training the classifier
clf.fit(X_train, y_train)

## Predicting the Species
predicted = clf.predict(X_test)

## Checking the accuracy
print(accuracy_score(predicted, y_test))