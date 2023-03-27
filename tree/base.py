"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

from audioop import avg
from posixpath import split
from time import sleep
from anyio import current_default_worker_thread_limiter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.feature_selection import SelectFdr
from sklearn.manifold import smacof
from sympy import N, re, solve, true
from .utils import entropy, information_gain, gini_index

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion="information_gain", max_depth=4, case=None):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > Default value of criterion is set as "information_gain", though criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        > Default value of max_depth is set to be 4
        """
        self.criterion=criterion
        self.max_depth=max_depth
        self.tree = {}
        self.case=case
        self.node_id=-1

    def split(self, X, y, max_depth, parent, criterion, case):
        # Base case is here
        if(max_depth==0 or len(y.unique())==1):
            # For discrete output return the mode value
            if(case=='dido' or case=='rido'):
                if(self.node_id>-1):
                    self.tree[parent]['children'].append(self.node_id+1)
                self.node_id+=1
                self.tree[self.node_id]={'attribute':y.mode()[0],'isleaf':True,'condition':[]}
                return
            
            if(case=='diro' or case=='riro'):
                if(self.node_id>-1):
                    self.tree[parent]['children'].append(self.node_id+1)
                self.node_id+=1
                self.tree[self.node_id]={'attribute':y.mean(),'isleaf':True,'condition':[]}
                return

        if(case == 'dido'):
            max_gain,best_attr=0,None
            # Loop to find the best attribute for splitting
            for label,content in X.items():
                if(criterion=='information_gain'):
                    gain=information_gain(y,content)
                    if(gain > max_gain):
                        max_gain,best_attr=gain,label
                if(criterion=="gini_index"):
                    # For gini index, that attribute is selected that minimizes the weighted average gini index after splitting over all possible values of that attribute.
                    split_gini,gini = gini_index(y),np.inf
                    # for label,content in X.items():
                    total_gini=0
                    for value in content.unique():
                        total_gini+=(y[X[label]==value].size/y.size) * gini_index(y[X[label]==value])
                    avg_gini=total_gini/len(content.unique())
                    if(split_gini > avg_gini):
                        split_gini,best_attr=avg_gini,label    
                    # If we are not able to find better gini index after split than the current one, then no need to split.
                    if(split_gini>=gini):
                        if(self.node_id>-1):
                            self.tree[parent]['children'].append(self.node_id+1)
                        self.node_id+=1
                        self.tree[self.node_id]={'attribute':y.mode()[0],'isleaf':True,'condition':[]}
                        return

            # Adding a new node for the best attribute
            if(best_attr==None):# No best attribute is found no end to split further from here
                if(self.node_id>-1):
                    self.tree[parent]['children'].append(self.node_id+1)
                self.node_id+=1
                self.tree[self.node_id]={'attribute':y.mode()[0],'isleaf':True,'condition':[]}
                return

            if(self.node_id>-1):
                self.tree[parent]['children'].append(self.node_id+1)
            self.node_id+=1
            curr_node=self.node_id
            self.tree[curr_node] = {'attribute':best_attr,'isleaf':False,'condition':X[best_attr].unique(),'children':[]}

            # Looping through each child of node. Recusively splitting at each child.
            for i in range(len(X[best_attr].unique())):
                valu = X[best_attr].unique()[i]
                self.split(X[X[best_attr]==valu], y[X[best_attr]==valu],max_depth-1, curr_node,criterion,case)
        
        if(case == 'diro'):
            best_attr,variance,reduction=None,y.var(),np.NINF
            # Finding the attribute that gives maximum reduction in weighted variance after split.
            for label,content in X.items():
                split_variance=0
                for valu in content.unique():
                    split_variance+= (y[X[label]==valu].size/y.size)*y[X[label]==valu].var()
                if(reduction < variance - split_variance):
                    reduction,best_attr=variance - split_variance,label
            
            if(self.node_id>-1):
                self.tree[parent]['children'].append(self.node_id+1)
            self.node_id+=1
            curr_node=self.node_id
            self.tree[curr_node] = {'attribute':best_attr,'isleaf':False, 
            'condition':X[best_attr].unique(),'children':[]}
            # Recursively splitting for all children.
            for i in range(len(X[best_attr].unique())):
                valu = X[best_attr].unique()[i]
                self.split(X[X[best_attr]==valu], y[X[best_attr]==valu], max_depth-1,curr_node, criterion,case)

        if(case == 'rido'):
            # Note : At a particular node, any attribute is splitted into two parts only i.e. a binary tree
            if(criterion=='information_gain'):
                best_attr = (None,0,-1) # Tuple for best attribute (attribute_name, gain value, index for split)
            if(criterion=='gini_index'):
                best_attr=(None, np.inf, -1)
            
            for label,content in X.items():
                Xy=X.copy()
                Xy.insert(X.shape[1],'y',y)
                Xy=Xy.sort_values(label)
                l,h=pd.Series(list(Xy['y'][0:Xy.shape[0]-1])), pd.Series(list(Xy['y'][1:]))
                same_value=list(l==h)
                if(criterion=='information_gain'):
                    local_best_split=(np.NINF,-1) # Tuple of (gain, index); index is the first element of the right half
                if(criterion=='gini_index'):
                    local_best_split=(np.inf,-1)

                
                for i in range(len(same_value)):
                    if(not same_value[i]):
                        # Comparison with the average value; based on comparision the attribute is discretized, so as to compute the gain.
                        temp_content =  Xy[label] > (Xy.iloc[i][label]+Xy.iloc[i+1][label])/2
                        if(criterion=='information_gain'):
                            curr_gain=information_gain(temp_content,Xy['y'])
                            if(local_best_split[0]<curr_gain):
                                local_best_split=(curr_gain, i+1)
                        if(criterion=='gini_index'):
                            gini = gini_index(Xy['y'])
                            # split_gini=(gini_index(Xy['y'][:i+1])+gini_index(Xy['y'][i+1:]))/2
                            split_gini=(temp_content[:i+1].size *gini_index(temp_content[:i+1])+temp_content[i+1:].size*gini_index(temp_content[i+1:]))/temp_content.size
                            if(split_gini < gini and split_gini<local_best_split[0]):
                                local_best_split = (split_gini,i+1)
                                
                                        
                if(criterion=='information_gain' and local_best_split[0]>best_attr[1]):
                    best_attr=(label, local_best_split[0],local_best_split[1])
                if(criterion=='gini_index' and local_best_split[0]<best_attr[1]):
                    best_attr=(label, local_best_split[0], local_best_split[1])

            if(best_attr==None):# No best attribute is found no end to split further from here
                if(self.node_id>-1):
                    self.tree[parent]['children'].append(self.node_id+1)
                self.node_id+=1
                self.tree[self.node_id]={'attribute':y.mode[0],'isleaf':True,'condition':[]}
                return 

            # BEST attribute and split index has been found :)
            Xy=X.copy()
            Xy.insert(X.shape[1],'y',y)
            Xy=Xy.sort_values(best_attr[0])
            # Adding a new node for the best attribute
            if(self.node_id>-1):
                self.tree[parent]['children'].append(self.node_id+1)
            self.node_id+=1
            curr_node=self.node_id
            split_index=best_attr[2]
            split_value = (list(Xy[best_attr[0]])[split_index - 1]+list(Xy[best_attr[0]])[split_index])/2
            self.tree[curr_node] = {'attribute':best_attr[0],'isleaf':False,'condition':split_value,'children':[]}

            # Looping through left and right subtree. Recusively splitting at each child.                
            self.split(Xy.iloc[:split_index,:-1],Xy.iloc[:split_index,-1],max_depth-1,curr_node,criterion,case)
            self.split(Xy.iloc[split_index:,:-1],Xy.iloc[split_index:,-1], max_depth-1,curr_node,criterion,case)
        
        if(case=='riro'):
            best_attr,split_ind,loss=None,-1,np.inf
            squared_loss = np.square(y-y.mean()).sum()
            loss=squared_loss
            for label,content in X.items():
                Xy=X.copy()
                Xy.insert(X.shape[1],'y',y)
                Xy=Xy.sort_values(label)
                #Compute loss for every possible split
                for i in range(1,X.shape[0]-1):
                    left_loss=np.square(Xy.iloc[:i]['y'] - Xy.iloc[:i]['y'].mean()).sum()
                    right_loss=np.square(Xy.iloc[i:]['y'] - Xy.iloc[i:]['y'].mean()).sum()
                    if(left_loss+right_loss < loss):
                        loss=left_loss+right_loss
                        best_attr,split_ind=label,i
            
            if(best_attr == None):
                # No need to split
                if(self.node_id>-1):
                    self.tree[parent]['children'].append(self.node_id)
                self.node_id+=1
                self.tree[self.node_id]={'attribute':y.mean(),'isleaf':True,'condition':[]}
                return
            
            if(self.node_id>-1):
                self.tree[parent]['children'].append(self.node_id+1)
            self.node_id+=1
            curr_node=self.node_id
            self.tree[curr_node]={'attribute':best_attr,'isleaf':False,
            'condition':Xy.iloc[:split_ind][best_attr].mean(),'children':[]}

            # Looping through left and right subtree. Recusively splitting at each child.                
            self.split(Xy.iloc[:split_ind,:-1],Xy.iloc[:split_ind,-1],max_depth-1,curr_node,criterion,case)
            self.split(Xy.iloc[split_ind:,:-1],Xy.iloc[split_ind:,-1],max_depth-1,curr_node,criterion,case)          

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        if(self.case==None) :
            if(y.dtype=='float' or y.dtype=='int'):
                if(X.dtypes[0]=='float' or X.dtypes[0]=='int'):
                    case='riro'
                else:
                    case='diro'
            elif(y.dtype!='float' or y.dtype!='int'):
                if(X.dtypes[0]=='float' or X.dtypes[0]=='int'):
                    case='rido'
                else:
                    case='dido'
            self.case=case
        else:
            case=self.case
        self.split(X,y,self.max_depth,self.node_id,self.criterion,case)
        pass

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        if(self.case=='dido' or self.case=='diro'):
            lis=[]
            for index,x in X.iterrows():
                root=0
                while(not self.tree[root]['isleaf']):
                    # Find which value is in X
                    val = x[self.tree[root]['attribute']]
                    child=np.where(self.tree[root]['condition'] == val)[0][0]
                    root=self.tree[root]['children'][child]
                if(self.tree[root]['isleaf']):
                    lis.append(self.tree[root]['attribute'])
            y=pd.Series(lis)
            return y

        if(self.case=='rido'):
            lis=[]
            for index,x in X.iterrows():
                root=0
                while(not self.tree[root]['isleaf']):
                    valu = x[self.tree[root]['attribute']]
                    if(valu <= self.tree[root]['condition']):
                        root=self.tree[root]['children'][0]
                    else:
                        root=self.tree[root]['children'][1]
                if(self.tree[root]['isleaf']):
                    lis.append(self.tree[root]['attribute'])
            y=pd.Series(lis)
            return y
        
        if(self.case=='riro'):
            lis=[]
            for index,x in X.iterrows():
                root,done=0,False
                visited={0}
                while(not (self.tree[root]['isleaf'] or done) ):
                    valu=x[self.tree[root]['attribute']]
                    if(valu <= self.tree[root]['condition']):
                        root=self.tree[root]['children'][0]

                        if(root in visited):
                            lis.append(self.tree[root]['condition'])
                            done=True
                        else:
                            visited.add(root)
                    else:
                        root=self.tree[root]['children'][1]
                if(self.tree[root]['isleaf']):
                    lis.append(self.tree[root]['attribute'])
            y=pd.Series(lis)
            return y
        pass


    def print_tree(self, node, level,visited):
        if(self.tree[node]['isleaf']):
            print('    '*(level+1)+str(self.tree[node]['attribute']))    

        if(not self.tree[node]['isleaf']):
            if(self.case == 'rido'):
                print('    '*(level+1)+str(self.tree[node]['attribute'])+' <= '+str(self.tree[node]['condition']))
                self.print_tree(self.tree[node]['children'][0], level+1,visited)
                print('    '*(level+1)+str(self.tree[node]['attribute'])+' > '+str(self.tree[node]['condition']))
                self.print_tree(self.tree[node]['children'][1], level+1,visited)
            
            if(self.case=='riro'):
                if(node not in visited):
                    visited.add(node)
                    print('    '*(level+1)+str(self.tree[node]['attribute'])+' <= '+str(self.tree[node]['condition']))
                    self.print_tree(self.tree[node]['children'][0], level+1,visited)
                    print('    '*(level+1)+str(self.tree[node]['attribute'])+' > '+str(self.tree[node]['condition']))
                    self.print_tree(self.tree[node]['children'][1], level+1,visited)
                else:
                    return
            
            if(self.case=='diro' or self.case=='dido'):
                print('    '*(level)+str(self.tree[node]['attribute']))
                for i in range(len(self.tree[node]['children'])):
                    child = self.tree[node]['children'][i]
                    print('    '*(level+1)+str(self.tree[node]['attribute'])+ ' = '+str(self.tree[node]['condition'][i]))
                    self.print_tree(child,level+2,visited)

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        visited=set()
        self.print_tree(0,0,visited)
        pass
