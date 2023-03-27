
from sklearn.feature_selection import SelectFdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.n_estimators = n_estimators
        self.classifiers = base_estimator
        self.X = None
        self.y = None
        pass

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X,self.y=X,y
        X_=X.copy()
        X_['y']=y
        for i in range(self.n_estimators):
            Data = X_.sample(n=X.shape[0], replace=True)
            X_train,y_train=Data.iloc[:,:-1],Data.iloc[:,-1]
            self.classifiers[i].fit(X_train,y_train)
        pass

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        results=[]
        for i in range(self.n_estimators):
            y_hat = self.classifiers[i].predict(X)
            y_hat=pd.Series(y_hat)
            results.append(y_hat)
        y_combined = pd.concat(results, axis=1)
        y_final = y_combined.mode(axis=1) # .mode return a DataFrame, which is converted to Sereis
        y_final = y_final[0]
        return y_final

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

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
        for clf in self.classifiers:

            ax1[cnt].set_xlim(x_min,x_max)
            ax1[cnt].set_ylim(y_min,y_max)
            ax2.set_xlim(x_min,x_max)
            ax2.set_ylim(y_min,y_max)
            
            ax1[cnt].set_aspect('equal', 'box')
            ax2.set_aspect('equal', 'box')

            ax1[cnt].scatter(self.X.iloc[:,0],self.X.iloc[:,1],c=self.y, edgecolors="k")

            # predict_proba is a method of class scikit learn DecisionTreeClassifier. It gives probablity for each class in the input.
            # The shape of Z is No. of samples in input X No. classes. 
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]) 
            Z=Z[:,1]  # Only the probablity of either classes is needed.
            Z = Z.reshape(xx.shape)
            ax1[cnt].contourf(xx,yy,Z,cmap=cm_bright, alpha=0.4)
            ax2.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.4)  
            ax2.scatter(self.X.iloc[:,0],self.X.iloc[:,1], c=self.y, edgecolors="k")
            
            
            title = "Round {}".format(cnt+1)
            ax1[cnt].set_title(title)
            ax2.set_title("Combined Decision Surface")
            cnt+=1
        
        fig1.set_size_inches(14, 8)
        fig2.set_size_inches(14, 8)
        fig1.tight_layout()
        fig2.tight_layout()
        return [fig1, fig2]
