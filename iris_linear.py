from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn import metrics
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test =  train_test_split(iris.data, iris.target, test_size=0.20, random_state=111)
crossvalidation = KFold(n=X_train.shape[0], n_folds=5,shuffle=True, random_state=1)

maxScore = 0.0;
maxReg = 0.0;

for regularization in np.arange(0.1, 10, 0.5):
	logClassifier = linear_model.LogisticRegression(C=150,random_state=111)
	
	score = np.mean(cross_val_score(logClassifier, X_train, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
	if(score > maxScore):
			maxScore = score
			maxReg= regularization	
	print 'Regularization:{0}  Accuracy: {1:.3f}'.format(regularization,score) #to identify the best regularization iteratively

#taking C = 150 as regularization
logClassifier = linear_model.LogisticRegression(C=150,random_state=111)
logClassifier.fit(X_train, y_train)
predicted = logClassifier.predict(X_test)
print predicted
print y_test
print metrics.accuracy_score(y_test, predicted)  # 1.0 is 100 percent accuracy

print maxReg

#taking C = 150 as regularization
logClassifier = linear_model.LogisticRegression(C=1,random_state=111)
logClassifier.fit(X_train, y_train)
predicted = logClassifier.predict(X_test)
print predicted
print y_test
print metrics.accuracy_score(y_test, predicted)  # 1.0 is 100 percent accuracy
