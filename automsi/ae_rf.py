
from sklearn.ensemble import RandomForestRegressor

import numpy as np

class PatchAdapter():
    
    def __init__(self, x, y, labels):
        self.x = x
        self.y = y
        self.labels = labels
        self.x_mean = None
        self.y_mean = None
        
    def reduce_to_mean(self, label):
        self.x_mean = np.mean(self.x, axis = (1,2))
        self.y_mean = np.mean(self.y[...,self.labels[label]], axis = (1,2))
        return self
    
    def undersample_non_label_patches(self, cutoff, other_fraction):
        patches_of_interest = np.where(self.y_mean >= cutoff)[0]
        len_poi = len(patches_of_interest)
        if other_fraction > 0.:        
            other_patches = np.where(self.y_mean < cutoff)[0]
            other_patches = np.random.choice(other_patches, min(int(len_poi * other_fraction), len(other_patches)), replace=False)
        else: 
            other_patches = patches_of_interest[0:1]
        print("Number of labelled patches for train/test: " + str(len_poi))
        print("Number of other patches: " + str(len(other_patches)))
        
        self.poi = np.union1d(patches_of_interest, other_patches)
        return self
    
    
    
    # n_bins of 5 may be too few for larger datasets.
    def create_bins(self, n_bins = 5):
        min_y = np.amin(self.y_mean)
        max_y = np.amax(self.y_mean)
        
        bins = np.linspace(start=min_y, stop=max_y, num=n_bins)
        y_binned = np.digitize(self.y_mean[self.poi], bins, right=True)
        
        for i in range(0, n_bins):
            print(len(np.where(y_binned == i)[0]))
            
        self.y_bins = y_binned
        return self

    
class RandomForestSummary:

    def __init__(self, config, 
                 patches, score, 
                 top_name, top_imp, top_occ):
        self.config = config
        self.patches = patches
        self.score = score
        self.top_name = top_name
        self.top_imp = top_imp
        self._top_occ = top_occ


    

class RandomForestRegressionBuilder:

    def __init__(self, n_estimators: int, max_features, random_state):
        self.forest = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=random_state)

    def fit(self, z_train, y_train, sample_weight):
        self.forest.fit(z_train, y_train, sample_weight)
        return self
    
    
class RandomForestClassificationBuilder:

    def __init__(self, n_estimators: int, max_features, random_state):
        self.forest = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=random_state)

    def fit(self, z_train, y_train, sample_weight):
        self.forest.fit(z_train, y_train, sample_weight)
        return self
    