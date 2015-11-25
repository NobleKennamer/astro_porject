"""
This file contains common utility functions
@author: Abhisaar Sharma
"""

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