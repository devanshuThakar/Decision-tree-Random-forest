
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

def combine(X1,X2, tree1, tree2, y):
    y_hat1=tree1.predict(X1)
    y_hat2=tree2.predict(X2)
    e1=rmse(y_hat1, y)
    e2=rmse(y_hat2,y)
    w1,w2=1/e1,1/e2
    return (w1*y_hat1+w2*y_hat2)/(w1+w2)

# Read real-estate data set
# ...
# 
df = pd.read_fwf("auto-mpg.data",names=["mpg", "cylinders","displacement","horsepower","weight","acceleration","model_year","origin",'car_name'])
print("The columns in dataframe are  : ", list(df.columns))

# Cleaning and conversion of data. 
# Removing the extra '' from car_name 
df['car_name']=df['car_name'].apply(lambda x:x[1:len(x)-1])
#horsepower attribute has a datatype of string converting it to numeric
df=df[df['horsepower']!='?']
df['horsepower']=pd.to_numeric(df['horsepower'], errors='coerce')

# df=df.dropna() # Dropping Nan values
X=df.iloc[:,:-1]
y=df.iloc[:,0]
# Resetting the index of X and y
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
print("The shape of X : ",X.shape, ". The size of y is : ", y.size)
print("The datatypes of columns after coversion are \n",df.dtypes)

# tree = DecisionTree(criterion="gini_index",case="rido")
# tree.fit(X, y)
# y_hat = tree.predict(X)

print("Q3-(a) Usuage of decision tree for automotive efficiency problem.")

print("Since the input data is mix of real and discrete data, Two type of decision trees were learned (1) For real input and (2) For discrete output. Final output (i.e. mpg) can be obtained as combination from both the trees.")

# For Real Input Real Output part of input
X1=X[["displacement","horsepower","weight","acceleration"]]

tree1 = DecisionTree(criterion="information_gain",max_depth=6,case="riro")
tree1.fit(X1, y)
y_hat1 = tree1.predict(X1)
print("\nThe training error for Real Input Real Output (RIRO) type decision tree having max-depth {} is : ".format(tree1.max_depth))
print('RMSE: ', rmse(y_hat1, y))
print('MAE: ', mae(y_hat1, y))


# For Discrete Input Real Output part of input
X2=X[["cylinders","model_year","origin"]]

tree2 = DecisionTree(criterion="information_gain",max_depth=6,case="diro")
tree2.fit(X, y)
y_hat2 = tree2.predict(X)
print("\nThe training error for Discrete Input Real Output (DIRO) type decision tree having max-depth {} is : ".format(tree2.max_depth))
print('RMSE: ', rmse(y_hat2, y))
print('MAE: ', mae(y_hat2, y))

print("\nQ3-(b) Comparison with scikit learn")
df3=df[["displacement","horsepower","weight","acceleration","cylinders","model_year","origin","mpg"]]

df3=df3.dropna() # Dropping Nan values

X=df3.iloc[:,:-1]
y=df3.iloc[:,-1]
y=pd.Series(list(y))
regressor = DecisionTreeRegressor(random_state=0, max_depth=6)
regressor.fit(X,y)
yhat=regressor.predict(X)
print("\nThe training error for Scikit-learn's decision (regression) tree having max-depth {} is : ".format(regressor.max_depth))
print('RMSE: ', rmse(yhat, y))
print('MAE: ', mae(yhat, y))

print("\nThe performance by combining both the decision trees (Real input and discrete input) having depth 6 learn in part-(a) is shown below : ")
y_hat = combine(X1,X2,tree1,tree2,y)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
