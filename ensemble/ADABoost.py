from email.mime import base
from random import sample
from time import sleep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.n_estimators = n_estimators
        self.classifiers=base_estimator
        self.alpha=[]
        self.weights=None
        self.X=None
        self.y = None

        pass

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        w = pd.Series([1/X.shape[0] for i in range(X.shape[0])])
        self.weights=[]
        for m in range(self.n_estimators):
            self.classifiers[m].fit(X,y,sample_weight=w)
            self.weights.append(w)
            y_hat = self.classifiers[m].predict(X)
            err = w[y!=y_hat].sum()/w.sum()
            alpha_m = 0.5*np.log((1-err)/err)
            self.alpha.append(alpha_m)
            w = w.multiply(np.exp(-alpha_m*(y==y_hat))) # Multiplication factor for correct prediction
            w = w.multiply(np.exp(alpha_m*(y!=y_hat)))  # Multiplication factor for incorrect prediction
            w = w/w.sum()
        self.X = X
        self.y = y
        pass

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        results = []
        y_final=pd.Series([0 for i in range(X.shape[0])])
        for i in range(self.n_estimators):
            y_hat = self.classifiers[i].predict(X)
            y_hat=pd.Series(y_hat)
            y_hat=y_hat.replace(to_replace=0, value=-1)
            results.append(y_hat)
            y_final += self.alpha[i]*y_hat
        y_final = np.sign(y_final)
        y_final=y_final.replace(to_replace=-1, value=0)
        return y_final

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        fig1,ax1 = plt.subplots(1,self.n_estimators)
        fig2,ax2 = plt.subplots()
        cnt=0
        cm_bright = ListedColormap(["#4B0082", "#FFD700"])
        x_min, x_max = self.X.iloc[:, 0].min() - 0.1, self.X.iloc[:, 0].max() + 0.1
        y_min, y_max = self.X.iloc[:, 1].min() - 0.1, self.X.iloc[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        for alph, clf in zip(self.alpha, self.classifiers):

            ax1[cnt].set_xlim(x_min,x_max)
            ax1[cnt].set_ylim(y_min,y_max)
            ax2.set_xlim(x_min,x_max)
            ax2.set_ylim(y_min,y_max)
            
            ax1[cnt].set_aspect('equal', 'box')
            ax2.set_aspect('equal', 'box')

            ax1[cnt].scatter(self.X.iloc[:,0],self.X.iloc[:,1], s=1000*self.weights[cnt],c=self.y, edgecolors="k")

            # predict_proba is a method of class scikit learn DecisionTreeClassifier. It gives probablity for each class in the input.
            # The shape of Z is No. of samples in input X No. classes. 
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]) 
            Z=Z[:,1]  # Only the probablity of either classes is needed.
            Z = Z.reshape(xx.shape)
            ax2.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.4)  
            ax2.scatter(self.X.iloc[:,0],self.X.iloc[:,1],c=self.y, edgecolors="k")
            
            
            title = "alpha = {0:.4f}".format(alph)
            ax1[cnt].set_title(title)
            ax2.set_title("Combined Decision Surface")
            cnt+=1
        
        fig1.set_size_inches(14, 8)
        fig2.set_size_inches(14, 8)
        fig1.tight_layout()
        fig2.tight_layout()

        return [fig1, fig2]
