import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

#iris dataset, 3 species of flowers, 50 samples each
iris = datasets.load_iris()

# view the 3 categories. referenced from scikit website
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# print a scatter plot in 3D (using the first three features)
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], iris.data[:, 2], c=iris.target)

ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])

ax.legend(*scatter.legend_elements(), title="Classes", loc="lower right")

plt.show()

# 2-feature SVM
X = iris.data[:, :2]  # only use the first two features for 2D visualization
y = iris.target

# binary classification
y = (y == 0).astype(int)  # set 0 for first species, 1 for Not first species

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train svm model
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

# create mesh grid to plot the decision boundary
h = .02  # Step size in mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# predict class labels for each point in the mesh grid
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# plot decision boundary 
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)

# plot support vectors
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], facecolors='none', edgecolors='k', s=100, label='Support Vectors')

# labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Linear Classifier with Decision Boundary')
plt.legend()

plt.show()