from abc import ABCMeta, abstractmethod
import numpy as np

"""
This file contains the Abstract Classifier model implemented by each classifier.
It also contains a DAO DataSource.
@author: abhisaar
"""

class AbstractClassifier():
    __metaclass__ = ABCMeta
    @abstractmethod
    def fit(self, data, label): 
        """
        Abstract method which will train the model for the given
        Data and corresponding labels 
        """
        return

    @abstractmethod
    def predict(self, data):
        """
        Abstract method which will return the predictions for a given
        testing data. 
        """
        return
   
class DataSource(object):
    """
    DAO Datasource that returns the data
    """  
    __PATH_LYRAE = './data/RR_Lyrae_data.npz' 
    __PATH_REDSHIFT = "./data/redshift_mags.npz" 
    __PATH_QUASAR_STAR = "./data/quasar_star.npz"    

    """
    getData loads a .npz file and returns key-value pairs including 
    'arr_0' - A four tuple feature attributes [..,..,..,..]
    'arr_1' - A binary Prediction label
    returnValue: two lists of training and prediction Data 
    """
    def getLyraeData(self):
        return np.load(DataSource.__PATH_LYRAE)['arr_0'], np.load(DataSource.__PATH_LYRAE)['arr_1'] 
    