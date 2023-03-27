"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

from email.mime import base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
# Or you could import sklearn DecisionTree
# from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################
   

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

print("Q5-a")
criteria = 'information_gain'
# tree = DecisionTree(criterion=criteria,max_depth=1)
tree = [DecisionTreeClassifier(criterion="entropy",max_depth=1) for i in range(n_estimators)]
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))
fig1.savefig("Images/Q5a-estimators.png")
fig2.savefig("Images/Q5a-DecisionSurface.png")


##### AdaBoostClassifier on Classification data set using the entire data set
print("\nQ5-b")

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)

# Shuffling the Data
print("Shuffling the data.\n")
data = np.column_stack((X,y))
np.random.shuffle(data)

# Splitting data for train and test
print("Splitting data for train and test.\n")
N=int(data.shape[0]*0.60)
X_train,y_train = pd.DataFrame(data[:N,:-1]),pd.Series(data[:N,-1])
X_test,y_test = pd.DataFrame(data[N:,:-1]),pd.Series(data[N:,-1])

print("\nPerformance of AdaBoostClassifier using 3 estimators\n")
criteria = 'information_gain'
# tree = DecisionTree(criterion=criteria,max_depth=1)
tree = [DecisionTreeClassifier(criterion="entropy",max_depth=1) for i in range(n_estimators)]
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=3 )
Classifier_AB.fit(X_train, y_train)
y_hat = Classifier_AB.predict(X_test)
[fig3, fig4] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))

fig3.savefig("Images/Q5b-estimators.png")
fig4.savefig("Images/Q5b-DecisionSurface.png")
# plt.show()

print("\nPerformance of decision stump (1-depth tree)\n")
Decision_stump=DecisionTreeClassifier(criterion="entropy", max_depth=1)
Decision_stump.fit(X_train, y_train)
y_hat = Decision_stump.predict(X_test)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))
