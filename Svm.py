import AbstractClasses
import Utils
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
import cvxopt.solvers
import numpy.linalg as la
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import math

"""
This class solves the classification problem using an SVM
@author: Abhisaar Sharma
"""

class linearSVM(AbstractClasses.AbstractClassifier):

    clf = svm.SVC(kernel='linear',class_weight='auto', probability=True)

    def fit(self, data, label):
        print ("linearSVM training started.")
        linearSVM.clf.fit(data, label)
        print("linearSVM trained")

    def predict(self, data):
        return linearSVM.clf.predict(data)


class polySVM(AbstractClasses.AbstractClassifier):

    clf = svm.SVC(kernel='poly',class_weight='balanced', probability=True)
    kernel='rbf'
    def fit(self, data, label):
        print ("polySVM training started.")
        polySVM.clf.fit(data, label)
        print("polySVM trained")

    def predict(self, data):
        return polySVM.clf.predict(data)

class sigmoidSVM(AbstractClasses.AbstractClassifier):

    clf = svm.SVC(kernel='sigmoid',class_weight='balanced')

    def fit(self, data, label):
        print ("sigmoidSVM training started.")
        sigmoidSVM.clf.fit(data, label)
        print("sigmoidSVM trained")

    def predict(self, data):
        return sigmoidSVM.clf.predict(data)

class rbSVM(AbstractClasses.AbstractClassifier):

    def __init__(self):
        self.clf = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)

    def fit(self, data, label):
        print ("rbSVM training started."),
        self.clf.fit(data, label)
        print("rbSVM trained")

    def predict(self, data):
        return self.clf.predict(data)

class cwSVM(AbstractClasses.AbstractClassifier):

    def __init__(self, class_weight = {0:0.2, 1:10.0}):
        self.class_weight = class_weight
        self.clf = svm.SVC(kernel='rbf', class_weight=self.class_weight, probability=True)

    def fit(self, data, label):
        print ("cwSVM " + str(self.class_weight) +" training started.")
        self.clf.fit(data, label)
        print("cwSVM trained")

    def predict(self, data):
        return self.clf.predict(data)    
    
################################################
#Custom implementation
################################################
#The custom implementation using cvxopt is unsuitable for large datasets
#SMO implementation of lagrange multipliers should be used - it should be run using pypy for faster computation
###############################################

class Kernel(object):
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            exponent = -np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2))
            return np.exp(exponent)
        return f

    @staticmethod
    def _polykernel(dimension, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
        return Kernel._polykernel(dimension=dimension, offset=1.0)

    @staticmethod
    def homogenous_polynomial(dimension):
        return Kernel._polykernel(dimension=dimension, offset=0.0)

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(kappa * np.dot(x, y) + c)
        return f

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SVMTrainer(object):
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def train(self, X, y):
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        print("called for " + str(n_samples) + "__"+ str(n_features))
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = \
            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self._kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)
   
    #This function computes the lagrange multipliers using the 
    #SMO algorithm.
    def _compute_multipliers(self, X, y):
        maxiter = 10000
        n_samples, n_features = X.shape
        tolerance = 0.3
        alphatol = math.exp(-4)
        alpha = [0] * n_samples
        b = 0
        passes = 0
        max_passes = 5
        iter = 0
        while (passes < max_passes and iter < maxiter):
            print("pass " + str(passes))
            num_changed_alphas = 0
            for i in range (n_samples):
                print(i)
                Ei = 0
                for j in range (n_samples): 
                    Ei = Ei + alpha[j] * y[j] * self._kernel(X[j], X[i])
                Ei = Ei - y[i] + b
                
                if((y[i]*Ei < -1 * tolerance and alpha[i] < self._c) or (y[i]*Ei > tolerance and alpha[i] > 0)):
                    j = random.randint(0, n_samples-1)
                    while (j == i):
                        j = random.randint(0, n_samples-1)
                    Ej = 0
                    for k in range (n_samples): 
                        Ej = Ej + alpha[k] * y[k] * self._kernel(X[j], X[k])
                    Ej = Ej - y[j] + b
                    alphaI = alpha[i]
                    alphaJ = alpha[j]
                    L = 0
                    H = self._c
                    if(y[i] == y[j]):
                        L = max(0, alphaI + alphaJ - self._c)
                        H = min(self._c, alphaI + alphaJ)
                    else:
                        L = max(0, alphaJ - alphaI)
                        H = min(self._c, self._c + alphaJ - alphaI)
                    
                    if(abs(L-H) < alphatol):
                        continue
                    eta = 2*self._kernel(X[i],X[j]) - self._kernel(X[i], X[i]) - self._kernel(X[j], X[j])
                    if(eta >= 0):
                        continue
                    
                    alphaJNew = alphaJ - y[j] * (Ei - Ej) / eta
                    if (alphaJNew > H):
                        alphaJNew = H
                    elif (alphaJNew < L):
                        alphaJNew = L
                    if (abs(alphaJNew - alphaJ) < alphatol):
                        continue
                    alpha[i] = alpha[i] + y[i] * y[j] * (alphaJ - alphaJNew)
                    alpha[j] = alphaJNew
                    
                    b1 = b - Ei - y[i] * (alpha[i] - alphaI) * self._kernel(X[i], X[i]) - y[j] * (alphaJNew - alphaJ) * self._kernel(X[i], X[j])
                    b2 = b - Ej - y[i] * (alpha[i] - alphaI) * self._kernel(X[i], X[j]) - y[j] * (alphaJNew - alphaJ) * self._kernel(X[j], X[j])
                    if(alpha[i] < self._c and alpha[i] > 0):
                        b = b1
                    elif(alpha[j] < self._c and alpha[j] > 0):
                        b = b2
                    else:
                        b = (b1 + b2) /2
                    num_changed_alphas = num_changed_alphas + 1
            
            if(num_changed_alphas == 0):
                passes = passes + 1
            else:
                print("Changing alpha")
                passes = 0
            iter = iter + 1
        if(passes==max_passes):
            print("passes equal max passes - cannot optimize")
            
        print("finished optimization")
        print sum(alpha)
        return alpha

    ##This functions computes the lagrange multipliers using cvxopt.
    def _compute_multipliers_cvxopt(self, X, y):
        n_samples, n_features = X.shape
        K = self._gram_matrix(X)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])  

class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)

    def predict(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        if (np.sign(result).item() == -1):
            return 0
        else: 
            return 1    


####################
#####Testing########
####################
'''
data, label = AbstractClasses.DataSource().getLyraeData()
trainingData, testingData, trainingLabel, testingLabel = train_test_split(data, label, test_size=0.25, random_state=7)
#modelObjects = [ linearSVM(), polySVM(), rbSVM(), cwSVM({0:1, 1:100})]
modelObjects = [ linearSVM(), cwSVM({0:1, 1:40})]

for classifier in modelObjects:
    classifier.fit(trainingData, trainingLabel)
    predictions = classifier.predict(testingData)
    Utils.perfMeasure(testingLabel.tolist(), predictions.tolist())
    fpr, tpr, thresholds_roc = metrics.roc_curve(testingLabel.tolist(), predictions.tolist(), pos_label=1)
    plt.plot(fpr, tpr, label = classifier.__class__.__name__)
'''    
'''        
for i in range (0,100,2):
    classifier = cwSVM({0:1, 1:i})  
    classifier.fit(trainingData, trainingLabel)      
    predictions = classifier.predict(testingData)
    Utils.perfMeasure(testingLabel.tolist(), predictions.tolist())
    fpr, tpr, thresholds_roc = metrics.roc_curve(testingLabel.tolist(), predictions.tolist(), pos_label=1)
    plt.plot(fpr, tpr, label = classifier.__class__.__name__ + ' rl:' + str(i))
      
        
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
'''