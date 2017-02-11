from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from itertools import cycle

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_tsne = TSNE(n_components = 2, learning_rate=100, random_state=10, verbose=2).fit_transform(X)

colors = cycle('rgbcmykw')
target_ids = range(len(target_names))
plt.figure()
for i, c, label in zip(target_ids, colors, target_names):
	plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1],c=c, label=label)
	plt.legend()

plt.show()