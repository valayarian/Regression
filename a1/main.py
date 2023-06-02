import os,sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import *
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import scipy.sparse

def gradientDescent(X_val,Y_val,X, Y, alpha=0.0029, maxNumIterations=200):
    m = np.shape(X)[0] #total samples
    n = np.shape(X)[1]  # total number of features = 2049
    #W = np.random.randn(n)
    W = np.zeros(n)
    cost_history_list = []
    cost_history_list_val = []
    reltol = 0.001
    diff = 0
    for current_iteration in range(maxNumIterations):  # begin the process
 
        # compute the dot product between our feature 'X' and weight 'W'
        y_estimated = X.dot(W)
 
        # calculate the difference between the actual and predicted value
        error = y_estimated - Y
 
        # calculate the cost (Mean squared error - MSE)
        cost_train = (1.0/np.shape(X)[0])*(np.linalg.norm(X.dot(W)-Y,2)**2) 
        
        # compute the dot product between our feature 'X' and weight 'W'
        y_estimated_val = X_val.dot(W)
 
        # calculate the difference between the actual and predicted value
        error_val = y_estimated_val - Y_val
 
        # calculate the cost (Mean squared error - MSE)
        cost_val = (1.0/np.shape(X_val)[0])*(np.linalg.norm(X_val.dot(W)-Y_val,2)**2)
        if(current_iteration > 0):
            diff = (cost_history_list_val[-1] - cost_val)/cost_history_list_val[-1]
            if(np.abs(diff)<reltol):
                cost_history_list.append(cost_train)
                cost_history_list_val.append(cost_val)
                break
        # Update our gradient by the dot product between
        # the transpose of 'X' and our error divided by the
        # total number of samples
        gradient = (1 / m) * X.T.dot(error)
 
        # Now we have to update our weights
        W = W - alpha * gradient
        cost_history_list.append(cost_train)
        cost_history_list_val.append(cost_val)
        
    return x_test.dot(W)

def least_squaresL2(x, y, Penalty_factor=5):
    xTx = x.T.dot(x)
    n = xTx.shape[0]
    xTx_mod = xTx + Penalty_factor*np.identity(n)
    xTx_inv = np.linalg.inv(xTx_mod)
    w = xTx_inv.dot(x.T.dot(y))
    return x_test.dot(w)

def Loss(w,x,y):
    m = x.shape[0] #number of samples
    h = scipy.sparse.csr_matrix((np.ones(y.shape[0]), (y, np.array(range(y.shape[0])))))
    h = np.array(h.todense()).T
    h=h[:,1:]
    scores = np.dot(x,w)
    scores -= np.max(scores)
    like = (np.exp(scores).T /np.sum(np.exp(scores),axis=1)).T
    #loss = (-1 / m) * np.sum(h * np.log(like)) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(h - like)) #And compute the gradient for that loss
    return grad

def classify(x_test,x_train,y_train):
    w = np.zeros([x_train.shape[1],len(np.unique(y_train))])
    iterat = 1000
    alpha = 0.00001
    for i in range(0,iterat):
        grad = Loss(w,x_train,y_train)
        w = w - (alpha * grad)
    s = np.dot(x_test,w)
    s-=np.max(s)
    a =(np.exp(s).T / np.sum(np.exp(s),axis=1)).T
    b = np.argmax(a,axis=1)
    return b 

def pretrain(train):
	x_train = train.copy()
    y_train = train[1]
    x_train.drop(x_train.iloc[:, 0:2], inplace=True, axis=1)
    x_train.insert(loc=0, column=1, value=1.0)
    for i in range(0, 2050):
        x_train.rename(columns={i: i-1}, inplace=True)
    return x_train,y_train

def pretest(test):
	x_test = test.copy()
	y_test = test[0]
	x_test.drop(x_test.iloc[:,0:1], inplace=True, axis=1 )
	x_test.insert(loc=0, column=0, value=1.0)
    return x_test,y_test

def preval(val):
	x_val = val.copy()
	y_val = val[1]
	x_val.drop(x_val.iloc[:, 0:2], inplace=True, axis=1)
	x_val.insert(loc=0, column=1, value=1.0)
	for i in range(0, 2050):
        x_val.rename(columns={i: i-1}, inplace=True)
    return x_val,y_val

def main(train_path, val_path, test_path, outpath, section):

	train = pd.read_csv(train_path,header=None)
	x_train, y_train = pretrain(train)

	test = pd.read_csv(test_path,header=None)
	x_test, y_test = pretest(test)

	val = pd.read_csv(val_path,header=None)
	x_val,y_val = preval(val)

	section=int(section)
	if section==1:
		Y_est = gradientDescent(x_val,x_val,x_train,y_train)

	else if section==2:
		Y_est = least_squaresL2(x_train,y_train,5)

	else if section==5:
		Y_est = classify(x_test,x_train,y_train)
		
	dict={'Sample_name': y_test, 'Output_score': Y_est}
	df=pd.DataFrame(dict, index=False)
	df.to_csv(outpath)


if _name_ == "_main_":
    main(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
