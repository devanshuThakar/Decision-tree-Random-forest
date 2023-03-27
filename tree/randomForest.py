import imp
from time import sleep
from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, n_features=2, use_scikit=False):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.max_depth=max_depth
        self.n_features=n_features
        self.use_scikit=use_scikit
        self.trees = [DecisionTree(criterion=self.criterion, max_depth=self.max_depth, case="rido") for i in range(self.n_estimators)]
        if(use_scikit):
            self.trees = [DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth) for i in range(self.n_estimators)]
        self.feature_list = []
        self.X = None
        self.y = None

        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X,self.y=X,y
        for i in range(self.n_estimators):
            X_train = X.sample(n=self.n_features,axis='columns')
            self.feature_list.append(list(X_train))
            self.trees[i].fit(X_train, y)
        
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        results=[]
        for i in range(self.n_estimators):
            y_hat = self.trees[i].predict(X[self.feature_list[i]])
            if(self.use_scikit):
                y_hat=pd.Series(y_hat)
            results.append(y_hat)
        y_combined = pd.concat(results, axis=1)
        y_final = y_combined.mode(axis=1) # .mode return a DataFrame, which is converted to Sereis
        y_final = y_final[0]
        return y_final

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        for i in range(self.n_estimators):
            plt.figure()
            plot_tree(self.trees[i], filled=True)
            plt.title("Tree "+str(i+1))
            plt.savefig("Images/Q7b_Tree_"+str(i+1)+".png")

        plot_colors = "rb"
        fig1,ax1 = plt.subplots(2,3)
        fig2,ax2 = plt.subplots()
        x_min, x_max = self.X.iloc[:, 0].min() - 0.1, self.X.iloc[:, 0].max() + 0.1
        y_min, y_max = self.X.iloc[:, 1].min() - 0.1, self.X.iloc[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        
        for pairidx, pair in enumerate([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]):
            
            ax1[pair[0],pair[1]].set_xlim(x_min,x_max)
            ax1[pair[0],pair[1]].set_ylim(y_min,y_max)
            
            title = "Tree {}".format(pairidx+1)
            ax1[pair[0],pair[1]].set_title(title)
            ax2.set_title("Combined Decision Surface")

            ax1[pair[0],pair[1]].set_aspect('equal', 'box')
            ax2.set_aspect('equal', 'box')
            ax1[pair[0],pair[1]].set_xlabel("X1")
            ax1[pair[0],pair[1]].set_ylabel("X2")
            ax2.set_xlabel("X1")
            ax2.set_ylabel("X2")

            Z = self.trees[pairidx].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = ax1[pair[0],pair[1]].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            ax2.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5)

            for i, color in zip(range(2), plot_colors):
                ax1[pair[0],pair[1]].scatter(
                    self.X[0][self.y==i],
                    self.X[1][self.y==i],
                    c=color,
                    edgecolor="black",
                    s=15)
                
                ax2.scatter(
                    self.X[0][self.y==i],
                    self.X[1][self.y==i],
                    c=color,
                    edgecolor="black",
                    s=15)
        fig1.set_size_inches(14, 8)
        fig2.set_size_inches(14, 8)
        fig1.tight_layout()
        fig2.tight_layout()
        return [fig1, fig2]

class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None, n_features=2, use_scikit=False):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.max_depth=max_depth
        self.n_features=n_features
        self.use_scikit=use_scikit
        self.trees = [DecisionTree(criterion=self.criterion, max_depth=self.max_depth, case="riro") for i in range(self.n_estimators)]
        if(use_scikit):
            self.trees = [DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth) for i in range(self.n_estimators)]
        self.feature_list = []
        self.X = None
        self.y = None
        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X,self.y=X,y
        for i in range(self.n_estimators):
            X_train = X.sample(n=self.n_features,axis='columns')
            self.feature_list.append(list(X_train))
            self.trees[i].fit(X_train, y)
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        results=[]
        for i in range(self.n_estimators):
            y_hat = self.trees[i].predict(X[self.feature_list[i]])
            if(self.use_scikit):
                y_hat=pd.Series(y_hat)
            results.append(y_hat)
        y_combined = pd.concat(results, axis=1)
        y_final = y_combined.mean(axis=1) # .mode return a DataFrame, which is converted to Sereis        
        # y_final = y_final
        return y_final

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        for i in range(self.n_estimators):
            plt.figure()
            plot_tree(self.trees[i], filled=True)
            plt.title("Tree "+str(i+1))
            plt.savefig("Images/Q7b_Tree_"+str(i+1)+".png")

        plot_colors = "rb"
        fig1,ax1 = plt.subplots(2,3)
        fig2,ax2 = plt.subplots()
        x_min, x_max = self.X.iloc[:, 0].min() - 0.1, self.X.iloc[:, 0].max() + 0.1
        y_min, y_max = self.X.iloc[:, 1].min() - 0.1, self.X.iloc[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        
        for pairidx, pair in enumerate([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]):
            
            ax1[pair[0],pair[1]].set_xlim(x_min,x_max)
            ax1[pair[0],pair[1]].set_ylim(y_min,y_max)
            
            title = "Tree {}".format(pairidx+1)
            ax1[pair[0],pair[1]].set_title(title)
            ax2.set_title("Combined Decision Surface")

            ax1[pair[0],pair[1]].set_aspect('equal', 'box')
            ax2.set_aspect('equal', 'box')
            ax1[pair[0],pair[1]].set_xlabel("X1")
            ax1[pair[0],pair[1]].set_ylabel("X2")
            ax2.set_xlabel("X1")
            ax2.set_ylabel("X2")

            Z = self.trees[pairidx].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = ax1[pair[0],pair[1]].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            ax2.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5)

            for i, color in zip(range(2), plot_colors):
                ax1[pair[0],pair[1]].scatter(
                    self.X[0][self.y==i],
                    self.X[1][self.y==i],
                    c=color,
                    edgecolor="black",
                    s=15)
                
                ax2.scatter(
                    self.X[0][self.y==i],
                    self.X[1][self.y==i],
                    c=color,
                    edgecolor="black",
                    s=15)
        fig1.set_size_inches(14, 8)
        fig2.set_size_inches(14, 8)
        fig1.tight_layout()
        fig2.tight_layout()
        return [fig1, fig2]
