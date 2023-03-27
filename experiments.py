
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100
# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
def make_plots(case, runtime, predict_time, N, M):
    runtime=np.array(runtime)
    predict_time=np.array(predict_time)

    if(case=="RIRO" or case=="RIDO"):
        which_N=3

        plt.scatter(M, runtime[which_N,:])
        m, b = np.polyfit(M, runtime[which_N,:], 1)
        plt.plot(np.array(M),m*np.array(M)+b)
        plt.title('Running time vs M for N={}'.format(N[which_N]))
        plt.xlabel('M')
        plt.legend(["theoretical",'expteriment'])
        plt.ylabel('Time(s)')
        plt.savefig('Images/'+case+"_Training_Time_vs_M_for_N={}.png".format(N[which_N]))
        plt.show()

    which_M=2

    plt.scatter(N, runtime[:,which_M])
    m, b = np.polyfit(N, runtime[:,which_M], 1)
    plt.plot(np.array(N),m*np.array(N)*np.log(np.array(N))+b)
    plt.title('Running time vs N for M={}'.format(M[which_M]))
    plt.legend(["theoretical",'expteriment'])
    plt.xlabel('N')
    plt.ylabel('Time(s)')
    plt.savefig('Images/'+case+"_Training_Time_vs_N_for_M={}.png".format(M[which_M]))
    plt.show()

    which_M=2

    plt.scatter(N, predict_time[:,which_M])
    m, b = np.polyfit(N, predict_time[:,which_M], 1)
    plt.plot(np.array(N),m*np.array(N)+b)
    plt.title('Prediction time vs N for M={}'.format(M[which_M]))
    plt.legend(["theoretical",'expteriment'])
    plt.xlabel('N')
    plt.ylabel('Time(ms)')
    plt.savefig('Images/'+case+"_Prediction_Time_vs_N_for_M={}.png".format(M[which_M]))
    plt.show()

# ..
# Function to create fake data (take inspiration from usage.py)
# ...
def create_data(case, n,m):
    if(case=="RIRO"):
        X = pd.DataFrame(np.random.randn(n,m))
        y = pd.Series(np.random.randn(n))
        return X,y

    if(case=="RIDO"):
        X = pd.DataFrame(np.random.randn(n,m))
        y = pd.Series(np.random.randint(2, size = n), dtype="category")
        return X,y
    
    if(case=="DIRO"):
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = n), dtype="category") for i in range(m)})
        y = pd.Series(np.random.randn(n))
        return X,y

    if(case=="DIDO"):
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = n), dtype="category") for i in range(m)})
        y = pd.Series(np.random.randint(10, size = n), dtype="category")
        return X,y

# ..other functions

cases = ['RIRO','RIDO','DIRO','DIDO']

for case in cases:
    N=[10,100,200,1000]
    M=[2,4,8,16,32]
    print("##################")
    print("For case of {}".format(case))
    runtime=[[0 for j in range(len(M))] for i in range(len(N))]
    predict_time=[[0 for j in range(len(M))] for i in range(len(N))]
    for i in range(len(N)):
        print("")
        n=N[i]
        for j in range(len(M)):
            m=M[j]
            X,y = create_data(case, n,m)

            tree=DecisionTree(criterion="information_gain")
            start = time.time()
            tree.fit(X,y)
            train_time=time.time()-start
            
            start=time.time()
            y_hat=tree.predict(X)
            pred_time=1000*(time.time()-start) # predict time in ms
            
            print("For N={} and M={} : Training time = {:.4f}s and Prediction time = {:.4f}ms".format(n,m,train_time,pred_time))
            runtime[i][j]=train_time
            predict_time[i][j]=pred_time
    make_plots(case,runtime,predict_time,N,M)
