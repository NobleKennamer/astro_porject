import scipy.optimize as opt
import numpy as np

"""
@author: Dylan Cockerham
"""

def logRegCost(theta, x, y):
    h = 1 / (1 + np.exp(- np.dot(x, theta)))
    cost = -y*np.log(h) - (1-y)*np.log(1-h)
    return cost.mean()

def logRegGrad(theta, x, y):
    h = 1 / (1 + np.exp(- np.dot(x, theta)))
    error = h - y
    grad = np.dot(error, x) / y.size
    return grad

def logRegFit(x_train, y_train):
    size, features = x_train.shape
    theta = 0.1* np.random.randn(features+1)
    x_mod = np.append( np.ones((size, 1)), x_train, axis=1)
    theta = opt.fmin_bfgs(logRegCost, theta, fprime=logRegGrad, args=(x_mod, y_train), disp=0)
    return theta

def logRegPredict(theta, x_test):
    x_mod = np.append( np.ones((x_test.shape[0], 1)), x_test, axis=1)
    y_pred = 1 / (1 + np.exp(- np.dot(x_mod, theta)))
    return np.round(y_pred)

def run_logistic_regression(x_train, y_train, x_test, y_test):
    theta = logRegFit(x_train, y_train)
    return logRegPredict(theta, x_test)