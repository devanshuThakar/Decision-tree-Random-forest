import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)
###Write code here

# Generating Dataset
from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Shuffling the Data
print("Shuffling the data.\n")
data = np.column_stack((X,y))
np.random.shuffle(data)

# Splitting data for train and test
print("Splitting data for train and test.\n")
N=int(data.shape[0]*0.60)
X_train,y_train = pd.DataFrame(data[:N,:-1]),pd.Series(data[:N,-1])
X_test,y_test = pd.DataFrame(data[N:,:-1]),pd.Series(data[N:,-1])

# Random Forest
criteria = "entropy"
Classifier_RF = RandomForestClassifier(n_estimators=6, criterion = criteria, max_depth=3, n_features=2, use_scikit=True)
Classifier_RF.fit(X_train, y_train)
y_hat = Classifier_RF.predict(X_test)
[fig1, fig2] = Classifier_RF.plot()
print('Criteria : ', criteria)
print('Accuracy : ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))

fig1.savefig("Images/Q7b-estimators.png")
fig2.savefig("Images/Q7b-DecisionSurface.png")
plt.show()