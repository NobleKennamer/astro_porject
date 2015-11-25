import AbstractClasses
import Utils
from sklearn import svm

"""
This class solves the classification problem using an SVM
@author: Abhisaar Sharma
"""


class SVM(AbstractClasses.AbstractClassifier):

    clf = svm.SVC()
    
    def fit(self, data, label):
        print ("SVM training started.")
        SVM.clf.fit(data, label)
        print("SVM trained")
        
    def predict(self, data):
        return SVM.clf.predict(data)
    
####################
#####Testing########
####################    
data, label = AbstractClasses.DataSource().getData()
supportVectorMachine = SVM()
supportVectorMachine.fit(data, label) 
predictions = supportVectorMachine.predict(data)
Utils.perfMeasure(label.tolist(), predictions.tolist())