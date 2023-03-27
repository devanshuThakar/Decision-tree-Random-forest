import numpy as np

def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    p=Y.value_counts()/Y.size
    return (-p.multiply(np.log2(p))).sum()

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    p=Y.value_counts()/Y.size
    return 1 - (p.multiply(p)).sum()

def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    Entropy_S=entropy(Y)
    sigma=0
    for value in attr.unique():
        Sv = Y[attr==value]
        sigma += Sv.size*entropy(Sv)
    return Entropy_S - (sigma/Y.size)
