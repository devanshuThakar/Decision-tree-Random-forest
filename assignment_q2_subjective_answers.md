### Cross Validation and Nested Cross Validation

On running `classification-exp.py`, the result is printed, which is shown below. 
```
Q2-part(a) 

Training the model on 70% data and testing on remaining 30%.

The accuracy for test dataset is 93.33333333333333%.
Precision for 0 is : 0.9166666666666666
Recall for 0 is : 0.9166666666666666
Precision for 1 is : 0.9444444444444444
Recall for 1 is : 0.9444444444444444

Q2-part(b)

Experiments for 5-Fold cross-validation.


For FOLD 1
With tree depth = 2
The accuracy for test dataset is 90.0%.
Precision for 1 is : 1.0
Recall for 1 is : 0.8
Precision for 0 is : 0.8333333333333334
Recall for 0 is : 1.0

For FOLD 2
With tree depth = 2
The accuracy for test dataset is 95.0%.
Precision for 0 is : 1.0
Recall for 0 is : 0.9333333333333333
Precision for 1 is : 0.8333333333333334
Recall for 1 is : 1.0

For FOLD 3
With tree depth = 2
The accuracy for test dataset is 90.0%.
Precision for 1 is : 1.0
Recall for 1 is : 0.8181818181818182
Precision for 0 is : 0.8181818181818182
Recall for 0 is : 1.0

For FOLD 4
With tree depth = 2
The accuracy for test dataset is 85.0%.
Precision for 1 is : 1.0
Recall for 1 is : 0.75
Precision for 0 is : 0.7272727272727273
Recall for 0 is : 1.0

For FOLD 5
With tree depth = 2
The accuracy for test dataset is 95.0%.
Precision for 1 is : 0.9230769230769231
Recall for 1 is : 1.0
Precision for 0 is : 1.0
Recall for 0 is : 0.875

Experiments to find optimum depth of the tree using 5-Fold Nested Cross-Validation.

First 80 smaples are used for training and validation. The last 20 samples are used for testing.

For FOLD 1
The size of training data :  (64, 2) 64
The size of validation data :  (16, 2) 16
With tree depth = 1
The accuracy for validation dataset is 93.75%.
With tree depth = 2
The accuracy for validation dataset is 93.75%.
With tree depth = 3
The accuracy for validation dataset is 81.25%.
With tree depth = 4
The accuracy for validation dataset is 87.5%.
With tree depth = 5
The accuracy for validation dataset is 81.25%.
With tree depth = 6
The accuracy for validation dataset is 81.25%.

For FOLD 2
The size of training data :  (64, 2) 64
The size of validation data :  (16, 2) 16
With tree depth = 1
The accuracy for validation dataset is 75.0%.
With tree depth = 2
The accuracy for validation dataset is 81.25%.
With tree depth = 3
The accuracy for validation dataset is 81.25%.
With tree depth = 4
The accuracy for validation dataset is 81.25%.
With tree depth = 5
The accuracy for validation dataset is 81.25%.
With tree depth = 6
The accuracy for validation dataset is 81.25%.

For FOLD 3
The size of training data :  (64, 2) 64
The size of validation data :  (16, 2) 16
With tree depth = 1
The accuracy for validation dataset is 93.75%.
With tree depth = 2
The accuracy for validation dataset is 93.75%.
With tree depth = 3
The accuracy for validation dataset is 93.75%.
With tree depth = 4
The accuracy for validation dataset is 93.75%.
With tree depth = 5
The accuracy for validation dataset is 93.75%.
With tree depth = 6
The accuracy for validation dataset is 93.75%.

For FOLD 4
The size of training data :  (64, 2) 64
The size of validation data :  (16, 2) 16
With tree depth = 1
The accuracy for validation dataset is 75.0%.
With tree depth = 2
The accuracy for validation dataset is 75.0%.
With tree depth = 3
The accuracy for validation dataset is 81.25%.
With tree depth = 4
The accuracy for validation dataset is 87.5%.
With tree depth = 5
The accuracy for validation dataset is 81.25%.
With tree depth = 6
The accuracy for validation dataset is 75.0%.

For FOLD 5
The size of training data :  (64, 2) 64
The size of validation data :  (16, 2) 16
With tree depth = 1
The accuracy for validation dataset is 87.5%.
With tree depth = 2
The accuracy for validation dataset is 87.5%.
With tree depth = 3
The accuracy for validation dataset is 81.25%.
With tree depth = 4
The accuracy for validation dataset is 81.25%.
With tree depth = 5
The accuracy for validation dataset is 81.25%.
With tree depth = 6
The accuracy for validation dataset is 81.25%.


The best model (having highest validation accuracy) has minimum depth of 1.
The highest validation accuracy is found as 93.75%.
The best model found out has a test accuracy of 95.0%.
```