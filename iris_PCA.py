from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle


iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
pca = PCA(n_components=2, whiten=True)
pca.fit(X)

print pca.components_


X_pca = pca.transform(X)

import numpy as np
np.round(X_pca.mean(axis=0), decimals=5)

np.round(X_pca.std(axis=0), decimals=5)


np.corrcoef(X_pca.T)

colors = cycle('rgbcmykw')
target_ids = range(len(target_names))
plt.figure()
for i, c, label in zip(target_ids, colors, target_names):
	plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],c=c, label=label)
	plt.legend()

plt.show()
