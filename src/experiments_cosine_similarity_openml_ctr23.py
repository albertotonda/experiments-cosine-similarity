# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:34:02 2023

First experiments with kPCA

@author: Alberto
"""

import matplotlib.pyplot as plt
import numpy as np
import openml
import seaborn as sns

from sklearn.decomposition import KernelPCA

# other state-of-the-art ensemble regressors
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def create_kpca_reference_points(n_samples : int) :
    """
    Creates a set of reference points in a high-dimensional space, that will be 
    later used to generate a cosine-similarity kPCA space. The original high-
    dimensional space will have size *n_samples*. For each sample, we will generate:
        - a reference point that has error 0.5 (and 0.0 elsewhere)
        - a reference point that has error 1.0 for that sample (and 0.0 elsewhere)
    And then a single reference point with error 0.0 for all sample

    Parameters
    ----------
    n_samples : int
        Number of samples in the original training set

    Returns
    -------
    reference_points : TYPE
        Reference points that will be used to create the kPCA space

    """
    
    # initially, generate reference points as a numpy array
    reference_points = np.zeros((n_samples*2 + 1, n_samples))
    
    # let's start with a first set of points in the form (0.0, ..., 1.0, 0.0, ..., 0.0)
    for i in range(0, n_samples) :
        for j in range(0, n_samples) :
            reference_points[i,j] = 1.0
            
    # and then another set with error 0.5
    for i in range(n_samples, 2*n_samples) :
        for j in range(0, n_samples) :
            reference_points[i,j] = 1.0
    
    return reference_points

if __name__ == "__main__" :
    
    # set a nice style for the plots
    sns.set_style()

    # load OpenML-CTR23, a regression benchmark suite with 35 benchmarks
    suite = openml.study.get_suite(353)
    print(suite)
    
    # suite.tasks contains the openml IDs for each of the regression tasks
    #for task_id in suite.tasks :
    #    task = openml.tasks.get_task(task_id)
    #    print(task)
    
    # let's focus on a specific task, for the moment
    # 361249 is 'white_wine', 4898 samples and 12 features
    task_ids = [361249]
    
    for task_id in task_ids :
        print("Now working on task #%d..." % task_id)
        
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        print("X:", X.shape)
        print("y:", y.shape)
        print("Column names:", attribute_names)
        
        print("Creating reference points...")
        reference_points = create_kpca_reference_points(y.shape[0])
        
        print("Computing cosine-similarity kPCA...")
        kpca = KernelPCA(n_components=2, kernel='cosine')
        X_pca = kpca.fit_transform(X)
        
        # let's create a first plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.scatter(X_pca[:,0], X_pca[:,1])
        
        plt.show()
        
        