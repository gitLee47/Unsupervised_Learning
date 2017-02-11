from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering  #Hierarchical Clustering which is implemented in scikit-learn as AgglomerativeClustering
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris.data
y = iris.target

#Predicting K using Silhouette
range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
	clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',linkage='ward')
	cluster_labels = clusterer.fit_predict(x)
	silhouette_avg = sm.silhouette_score(x, cluster_labels)
	print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)

# K Means Cluster
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(x)

predY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
print (model.labels_)
print (predY)
print iris.target


# Performance Metrics
print sm.accuracy_score(y, predY)

	
# Confusion Matrix
print sm.confusion_matrix(y, predY)