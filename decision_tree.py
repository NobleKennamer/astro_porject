import AbstractClasses
import Utils
from math import log
import numpy as np
from pandas import cut #used to discretize data
import operator

"""
This class solves the classification problem using Decision Tree
@author: Ariel Kruger
"""


def frequency(array):
    values, array = np.unique(array, return_inverse=True)
    return values, np.bincount(array.ravel())
 
class DecisionTreeClassifier(AbstractClasses.AbstractClassifier):   
    
    def _discretizeData(self, data, n_bins = 10):
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
    
    def _featureToSplit(self, dataset):
        n_features = len(dataset[0])-1
        baseEntropy = self._entropy(dataset)
        bestInfoGain = 0.0
        chosen_feature = -1
        for i in range(n_features):
            uniqueValues = set([row[i] for row in dataset])
            newEntropy = 0.0
            for value in uniqueValues:
                subDataset = self._splitData(dataset, i, value)
                prob=len(subDataset)/float(len(dataset))
                newEntropy += prob * self._entropy(subDataset)
            
            #calculate information gain
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                chosen_feature = i
                
        return chosen_feature
    
    def _buildTree(self, data,labels):
        
        list_of_classes = [row[-1] for row in data]
        
        #if it is a leaf node then count what is most probable class to predict
        if(len(list_of_classes) == 1): return self._mostProbableLabel(list_of_classes)
        
        if list_of_classes.count(list_of_classes[0]) == len(list_of_classes): return list_of_classes[0]
        
        feat_to_split = self._featureToSplit(data)
        feature_label = labels[feat_to_split]
        
        #set chosen feature as root of tree
        tree = {feature_label:{}}
        del(labels[feat_to_split])
        feature_values = [row[feat_to_split] for row in data]
        uniqueVals = set(feature_values)
        
        #recursively build tree 
        for value in uniqueVals:
            temp_labels = labels[:]
            tree[feature_label][value] = self._buildTree(self._splitData(data, feat_to_split, value), temp_labels)
        
        return tree
    
    def _classify(self, tree, data):
        firstStr = tree.keys()[0]
        
        subTrees = tree[firstStr]
        
        index_feature = self.labels.index(firstStr)
        
        return_class = 1 #return 1 as default
        
        for key in subTrees.keys():
            if data[index_feature] == key:
                if type(subTrees[key]).__name__ == 'dict':
                    return_class = self._classify(subTrees[key], data)
                else: return_class = subTrees[key]
        
        return return_class
    
    def fit(self, data, label):
        d_data = self._discretizeData(data, 150)
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
    data, labels = AbstractClasses.DataSource().getData()

    decision = DecisionTreeClassifier()
    decision.fit(data, labels) 
    predictions = decision.predict(data)
    Utils.perfMeasure(labels.tolist(), predictions.tolist())

