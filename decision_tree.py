import AbstractClasses
import Utils
from math import log
import numpy as np
from pandas import cut #used to discretize data
import operator
from sklearn.cross_validation import train_test_split

"""
This class solves the classification problem using Decision Tree
@author: Ariel Kruger
"""


def frequency(array):
    values, array = np.unique(array, return_inverse=True)
    return values, np.bincount(array.ravel())
 
class DecisionTreeClassifier(AbstractClasses.AbstractClassifier):
    
    def __init__(self, max_depth = 150):
        self.max_depth = max_depth
    
    def _discretizeData(self, data, n_bins):
        new_data = []
        self.bins = []
        for i in range(data.shape[1]):
            X = data.T[i]
            self.bins.append([])
            d_data, self.bins[i] = cut(X, n_bins, labels=False, retbins=True)
            new_data.append(d_data)

        return np.array(new_data).T
    
    def _discretizeDataBins(self, data):
        new_data = []
        for i in range(data.shape[1]):
            X = data.T[i]
            d_data = cut(X, bins = self.bins[i], labels=False, retbins=False)
            new_data.append(d_data)

        return np.array(new_data).T
    
    def _mostProbableLabel(self, list_labels):
        
        label_counter={}
        
        for label in list_labels:
            if label not in label_counter.keys():
                label_counter[label]=0
            label_counter[label] += 1
            
        sortedClassCount = sorted(label_counter.iteritems(), key=operator.itemgetter(1), reverse=True)
        
        return sortedClassCount[0][0]
    
    def _entropy(self, labels):
        """ Computes entropy used to compute the information gain of the features """
        n_labels = len(labels)

        if n_labels <= 1:
            return 0.0

        values, frequencies = frequency(labels)
        p = frequencies / float(n_labels)

        """ Compute entropy """
        entropy = 0.
        for i in p:
            entropy -= i * log(i, 2)

        return entropy
    
    def _splitData(self, data, n_col, value):
        data_to_return = []
        for d in data:
            if d[n_col] == value:
                temp_d = d[:n_col]
                temp_d.extend(d[n_col+1:])
                data_to_return.append(temp_d)
        
        return data_to_return
    
    def _featureToSplit(self, data):
        n_features = len(data[0])-1
        baseEntropy = self._entropy(data)
        bestInfoGain = 0.0
        chosen_feature = -1
        for i in range(n_features):
            uniqueValues = set([row[i] for row in data])
            newEntropy = 0.0
            for value in uniqueValues:
                subDataset = self._splitData(data, i, value)
                prob=len(subDataset)/float(len(data))
                newEntropy += prob * self._entropy(subDataset)
            
            #calculate information gain
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                chosen_feature = i
                
        return chosen_feature
    
    def _buildTree(self, data,labels, depth = 0):
        
        list_of_classes = [row[-1] for row in data]
        
        if list_of_classes.count(list_of_classes[0]) == len(list_of_classes): 
            return list_of_classes[0]
        
        #if it is a leaf node then count what is most probable class to predict
        if((len(list_of_classes) == 1)): 
            return self._mostProbableLabel(list_of_classes)
        
        '''if(self.max_depth != -1 and depth == self.max_depth):
            return self._mostProbableLabel(list_of_classes)'''
        
        depth = depth + 1
                
        feat_to_split = self._featureToSplit(data)
        try:
            feature_label = labels[feat_to_split]
        except:
            return self._mostProbableLabel(list_of_classes)
        
        #set chosen feature as root of tree
        tree = {feature_label:{}}
        del(labels[feat_to_split])
        feature_values = [row[feat_to_split] for row in data]
        uniqueVals = set(feature_values)
        
        #recursively build tree 
        for value in uniqueVals:
            temp_labels = labels[:]
            tree[feature_label][value] = self._buildTree(self._splitData(data, feat_to_split, value), temp_labels, depth)
        
        return tree
    
    def _classify(self, tree, data):
        firstStr = tree.keys()[0]
        
        subTrees = tree[firstStr]
        
        index_feature = self.labels.index(firstStr)
        
        return_class = 0 #return 1 as default
        
        for key in subTrees.keys():
            if data[index_feature] == key:
                if type(subTrees[key]).__name__ == 'dict':
                    return_class = self._classify(subTrees[key], data)
                else: return_class = subTrees[key]
        
        return return_class
    
    def fit(self, data, label):
        d_data = self._discretizeData(data, self.max_depth)
        d_data = np.column_stack((d_data,label)).tolist()
    
        nfeature=len(d_data[0])
        labels = [i for i in range(nfeature-1)]
        self.labels = [i for i in range(nfeature-1)]
    
        self.tree = self._buildTree(d_data,labels)
        
        
    def predict(self, data):
        #discretize test data accordingly to the bins used to discretize data when bulding tree
        d_data = self._discretizeDataBins(data) 

        result = []
        for d in d_data:
            result.append(self._classify(self.tree, d))
        
        return np.array(result)

if __name__ == "__main__":    
    data, labels = AbstractClasses.DataSource().getLyraeData()

    decision = DecisionTreeClassifier(max_depth = 8)
    
    
    colors_train, colors_test, labels_train, labels_test = train_test_split(data, labels, 
                                                                        test_size=0.25, random_state=13)
    
    decision.fit(colors_train, labels_train) 
    predictions = decision.predict(colors_test)
    
    Utils.perfMeasure(labels_test.tolist(), predictions.tolist())

