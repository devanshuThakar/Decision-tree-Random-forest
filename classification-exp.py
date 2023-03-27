import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read dataset
# ...
# 
from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("Images/Classification_Dataset.png")

# Q2-part(a)
print("Q2-part(a) \n")
print("Training the model on 70% data and testing on remaining 30%.\n")
X_train=pd.DataFrame(X[:70,:])
y_train=pd.Series(y[:70])

X_test=pd.DataFrame(X[70:,:])
y_test=pd.Series(y[70:])

tree = DecisionTree(criterion='information_gain',max_depth=2, case='rido')
tree.fit(X_train,y_train)

y_hat=tree.predict(X_test)

print('The accuracy for test dataset is '+str(accuracy(y_hat,y_test)*100)+'%.')
for cls in y_test.unique():
    print('Precision for '+str(cls)+' is : '+ str(precision(y_hat, y_test, cls)))
    print('Recall for '+str(cls)+' is : '+str(recall(y_hat, y_test, cls)))

# Q2-part(b)
print("\nQ2-part(b) \n")
print("Experiments for 5-Fold cross-validation.\n")

K=5
N=X.shape[0]
split=[(N)//5*(i) for i in range(0,K+1)]

for i in range(1,len(split)):    
    l,h=split[i-1],split[i]
    X_train=pd.DataFrame(np.vstack((X[:l,:],X[h:,:])))
    y_train=pd.Series(np.append(y[:l],y[h:]))
    
    X_test=pd.DataFrame(X[l:h,:])
    y_test=pd.Series(y[l:h])
    
    tree= DecisionTree(criterion='information_gain',max_depth=2, case='rido')
    tree.fit(X_train,y_train)
    
    y_hat=tree.predict(X_test)
    print("\nFor FOLD "+str(i))
    print("With tree depth = "+str(tree.max_depth))
    print('The accuracy for test dataset is '+str(accuracy(y_hat,y_test)*100)+'%.')
    for cls in y_test.unique():
        print('Precision for '+str(cls)+' is : '+ str(precision(y_hat, y_test, cls)))
        print('Recall for '+str(cls)+' is : '+str(recall(y_hat, y_test, cls)))

print('\nExperiments to find optimum depth of the tree using 5-Fold Nested Cross-Validation.\n')

print("First 80 smaples are used for training and validation. The last 20 samples are used for testing.")
K=5
N=80
split=[(N)//5*(i) for i in range(0,K+1)]
depth=[1,2,3,4,5,6]

X_test=pd.DataFrame(X[80:,:])
y_test=pd.Series(y[80:])

best_tree = (None, 0)

for i in range(1,len(split)):    
    l,h=split[i-1],split[i]
    print("\nFor FOLD "+str(i))
    
    X_train=pd.DataFrame(np.vstack((X[:l,:],X[h:N,:])))
    y_train=pd.Series(np.append(y[:l],y[h:N]))
    
    print('The size of training data : ', X_train.shape,y_train.size)
    X_validation=pd.DataFrame(X[l:h,:])
    y_validation=pd.Series(y[l:h])
    print('The size of validation data : ', X_validation.shape,y_validation.size)
    
    
    
    for d in depth:
        tree= DecisionTree(criterion='information_gain',max_depth=d, case='rido')
        tree.fit(X_train,y_train)
    
        y_hat=tree.predict(X_validation)
    
        print("With tree depth = "+str(tree.max_depth))
        accu=accuracy(y_hat,y_validation)
        if(best_tree[1]<accu):
            best_tree=(tree,accu)
        print('The accuracy for validation dataset is '+str(accu*100)+'%.')

best_model=best_tree[0]
print("\n\nThe best model (having highest validation accuracy) has minimum depth of "+str(best_model.max_depth)+'.')
print("The highest validation accuracy is found as "+str(best_tree[1]*100)+"%.")

y_hat=best_model.predict(X_test)
test_accuracy=accuracy(y_hat, y_test)
print("The best model found out has a test accuracy of "+str(test_accuracy*100)+"%.")