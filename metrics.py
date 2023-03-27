import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    # TODO: Write here
    try:
        assert(y_hat.size == y.size)
        return sum(y_hat==y)/y_hat.size
    except AssertionError:
        print("Sizes do not match.")
        pass
    

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """

    try:
        assert(y_hat.size == y.size)
        TP,FP=0,0
        for i in range(y_hat.size):
            if(y_hat[i]==cls and y[i]==cls):
                TP+=1
            elif(y_hat[i]==cls):
                FP+=1
        if(TP+FP==0):
            return 0
        else:
            return TP/(TP+FP)
    except AssertionError:
        print("Sizes do not match.")
        pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """

    try:
        assert(y_hat.size==y.size)
        TP,FN=0,0
        for i in range(y_hat.size):
            if(y_hat[i]==cls and y[i]==cls):
                TP+=1
            if(y_hat[i]!=cls and y[i]==cls):
                FN+=1
        if(TP+FN==0):
            return 0
        else:
            return TP/(TP+FN)
    except AssertionError:
        print("Sizes do not match.")
        pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    try:
        assert(y_hat.size == y.size)
        return np.sqrt(sum((y-y_hat)**2)/y_hat.size)
    except AssertionError:
        print("Sizes do not match.")
        pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    try:
        assert(y_hat.size == y.size)
        return (abs(y-y_hat).sum()/y_hat.size)
    except AssertionError:
        print("Sizes do not match.")
        pass