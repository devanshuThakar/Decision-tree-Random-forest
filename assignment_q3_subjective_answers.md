### Automotive efficiency 

<p><a href="https://archive.ics.uci.edu/ml/datasets/auto+mpg">  Automotive efficiency</a> problem is solved using scikit learn decison tree and the decision tree developed in Q1. 

On running `auto-efficiency.py`, the result is printed, which is shown below. 

```
The columns in dataframe are  :  ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
'acceleration', 'model_year', 'origin', 'car_name']
The shape of X :  (392, 8) . The size of y is :  392
The datatypes of columns after coversion are
 mpg             float64
cylinders         int64
displacement    float64
horsepower      float64
weight          float64
acceleration    float64
model_year        int64
origin            int64
car_name         object
dtype: object
Q3-(a) Usuage of decision tree for automotive efficiency problem.
Since the input data is mix of real and discrete data, Two type of decision trees 
were learned (1) For real input and (2) For discrete output. Final output (i.e. mpg) 
can be obtained as combination from both the trees.

The training error for Real Input Real Output (RIRO) type decision tree having max-depth 6 is :
RMSE:  15.032645618033587
MAE:  14.049795918367348

The training error for Discrete Input Real Output (DIRO) type decision tree having max-depth 6 is : 
RMSE:  3.341710728676004
MAE:  2.5444350139395904

Q3-(b) Comparison with scikit learn

The training error for Scikit-learn's decision (regression) tree having max-depth 6 is :
RMSE:  1.8372633689388513
MAE:  1.3030166788995607

The performance by combining both the decision trees (Real input and discrete input) having 
depth 6 learn in part-(a) is shown below :
RMSE:  4.297558242303572
MAE:  3.4082256064381227
```