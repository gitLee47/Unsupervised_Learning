#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
features = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=111)
crossvalidation = KFold(n=X_train.shape[0], n_folds=5,shuffle=True, random_state=1)

print X_train.shape
maxScore = 0.0;
maxDepth = 0;

for depth in range(1,10):
	tree_classifier = tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
	if tree_classifier.fit(X_train, y_train).tree_.max_depth < depth:
		break
	score = np.mean(cross_val_score(tree_classifier, X_train, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
	if(score > maxScore):
			maxScore = score
			maxDepth = depth	
	print 'Depth:{0}  Accuracy: {1:.3f}'.format(depth,score) #to identify the best depth iteratively

tree_classifier = tree.DecisionTreeClassifier(max_depth=maxDepth, random_state=0)
tree_classifier.fit(X_train,y_train)

predicted = tree_classifier.predict(X_test)

print predicted
print y_test

## Checking the accuracy
print(accuracy_score(predicted, y_test))


