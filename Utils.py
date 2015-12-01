"""
This file contains common utility functions
@author: Abhisaar Sharma
"""
import sklearn.metrics
import numpy as np

def perfMeasure(y_actual, y_hat, printResults=True):
    """
    This function returns and/or prints the performance metrics
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_hat)): 
        if int(y_actual[i]) == int(y_hat[i]) == 1:
            TP += 1
        if int(y_actual[i]) == 1 and int(y_hat[i]) == 0:
            FP += 1
        if int(y_actual[i]) == int(y_hat[i]) == 0:
            TN += 1
        if int(y_actual[i]) == 0 and int(y_hat[i]) == 1:
            FN += 1
    if(printResults):
        print("True Positives " + str(TP))
        print("False Positives " + str(FP))
        print("True Negatives " + str(TN))
        print("False Negatives " + str(FN))
    return(TP, FP, TN, FN)

'''
@author: Nobel
'''
def compute_results(y_true, y_pred):
    fpr, tpr, thresholds_roc = sklearn.metrics.roc_curve(y_true, y_pred, pos_label=1)
    precision_p, recall_p, thresholds_pr = sklearn.metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
    
    y_hard_decisions = np.round(y_pred)
    
    precision = sklearn.metrics.precision_score(y_true, y_hard_decisions)
    recall = sklearn.metrics.recall_score(y_true, y_hard_decisions)
    
    f1 = sklearn.metrics.f1_score(y_true, y_hard_decisions)
    roc_score = sklearn.metrics.roc_auc_score(y_true, y_hard_decisions)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_hard_decisions)
    
    results = {'roc': [fpr, tpr, thresholds_roc],
               'pr': [precision_p, recall_p, thresholds_pr],
               'scores': [precision, recall, f1, roc_score, accuracy]}
    return results