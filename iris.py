#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data[:-1,:], iris.target[:-1]

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X,y)

print "Predicted class , real class"
print logistic.predict(iris.data[-1,:])
print iris.target[-1]

print "Probabilities for each class from 0 to 2:" 
print logistic.predict_proba(iris.data[-1,:])
