# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:34:02 2023

First experiments with kPCA

@author: Alberto
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import seaborn as sns

from sklearn.decomposition import KernelPCA
from sklearn.metrics import r2_score

# state-of-the-art ensemble regressors
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

def create_kpca_reference_points(n_samples : int) :
    """
    Creates a set of reference points in a high-dimensional space, that will be 
    later used to generate a cosine-similarity kPCA space. The original high-
    dimensional space will have size *n_samples*. For each sample, we will generate:
        - a reference point that has error +/- 0.5 (and 0.0 elsewhere)
        - a reference point that has error +/- 1.0 for that sample (and 0.0 elsewhere)
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
    reference_points = np.zeros((n_samples*4 + 1, n_samples))
    
    # let's start with a first set of points in the form (0.0, ..., 1.0, 0.0, ..., 0.0)
    for i in range(0, n_samples) :
        for j in range(0, n_samples) :
            reference_points[i, j] = 1.0
            reference_points[n_samples+i, j] = -1.0
            reference_points[2*n_samples+i, j] = 0.5
            reference_points[3*n_samples+i, j] = -0.5
            
    return reference_points

def get_point_in_semantic_space(predictor, X, y_true) :
    
    y_pred = predictor.predict(X)

    return np.subtract(y_true, y_pred)

def get_points_in_semantic_space_from_predictors(predictors, X, y_true) :
    
    semantic_points = np.zeros((len(predictors), y_true.shape[0]))
    
    for i, predictor in enumerate(predictors) :
        semantic_points[i,:] = get_point_in_semantic_space(predictor, X, y_true)
        
    return semantic_points

if __name__ == "__main__" :
    
    # set a nice style for the plots (hopefully)
    sns.set_style()
    
    # hard-coded values
    random_seed = 42
    n_estimators = 100
    
    # set name for the folder where we will save the experiments
    main_folder = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_experiment_cosine_similarity")

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
    
    # set up main folder
    if not os.path.exists(main_folder) :
        os.makedirs(main_folder)
    
    for task_id in task_ids :
        print("Now working on task #%d..." % task_id)
        task_folder = os.path.join(main_folder, str(task_id))
        if not os.path.exists(task_folder) :
            os.makedirs(task_folder)
        
        # we also get the indices of the first fold, to use them as a measure of effectiveness
        task = openml.tasks.get_task(task_id)
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
        
        dataset = task.get_dataset()
        X_df, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        X = X_df.values
        print("X:", X)
        print("y:", y.shape)
        print("Column names:", attribute_names)
        
        # get the split
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        print("Creating reference points...")
        #reference_points = create_kpca_reference_points(y.shape[0])
        rng = np.random.default_rng(seed=random_seed)
        reference_points = rng.standard_normal(size=(X_train.shape[0], X_train.shape[0]))
        
        print("Computing cosine-similarity kPCA...")
        kpca = KernelPCA(n_components=2, kernel='cosine', random_state=random_seed)
        rp_pca = kpca.fit_transform(reference_points)
        
        # let's create a first plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.scatter(rp_pca[:,0], rp_pca[:,1], marker='.', alpha=0.3)
        
        ax.set_title("Cosine-similarity kPCA space with randomly generated errors")
        ax.set_xlabel("kPCA dimension 1")
        ax.set_ylabel("kPCA dimension 2")
        
        plt.savefig(os.path.join(task_folder, "kpca_random_values.png"), dpi=300)
        plt.close(fig)
        
        # and now we go more in depth, analyzing how each regressor generates its trees
        # (weak predictors) inside the kPCA space we just created
        # TODO  we might consider having a train/test split, but does it matter?
        #       well, it might matter to have an assessment of performance
        
        
        # let's start with good ol' Random Forest
        print("Now working with Random Forest...")
        regressor = RandomForestRegressor(random_state=random_seed)
        regressor.fit(X_train, y_train)
        y_test_pred = regressor.predict(X_test)
        performance = r2_score(y_test, y_test_pred)
        
        # now, we test each tree on the training data, to get the errors with respect to each sample
        rf_semantic_points = get_points_in_semantic_space_from_predictors(regressor.estimators_, X_train, y_train)
        rf_behavior_points = kpca.transform(rf_semantic_points)
        
        # another plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.scatter(rf_behavior_points[:,0], rf_behavior_points[:,1], marker='.', alpha=0.3, color='green')
        
        ax.set_title("Random Forest predictors (%d decision trees), R2=%.4f" % (len(regressor.estimators_), performance))
        ax.set_xlabel("kPCA dimension 1")
        ax.set_ylabel("kPCA dimension 2")
        
        plt.savefig(os.path.join(task_folder, "kpca_random_forest_predictors.png"), dpi=300)
        plt.close(fig)
        
        # let's proceed on to Gradient Boosting
        print("Now working with Gradient Boosting...")
        regressor = GradientBoostingRegressor(random_state=random_seed)
        regressor.fit(X_train, y_train)
        y_test_pred = regressor.predict(X_test)
        performance = r2_score(y_test, y_test_pred)
        
        gb_semantic_points = get_points_in_semantic_space_from_predictors(
            [e[0] for e in regressor.estimators_], X_train, y_train
            )
        gb_behavior_points = kpca.transform(gb_semantic_points)
        
        # and another plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for i in range(0, gb_behavior_points.shape[0]) :
            ax.scatter(gb_behavior_points[i,0], gb_behavior_points[i,1], marker="$%d$" % i, alpha=0.3, color='orange')
        
        ax.set_title("Gradient Boosting predictors (%d predictors), R2=%.4f" % (len(regressor.estimators_), performance))
        ax.set_xlabel("kPCA dimension 1")
        ax.set_ylabel("kPCA dimension 2")
        
        plt.savefig(os.path.join(task_folder, "kpca_gradient_boosting_predictors.png"), dpi=300)
        plt.close(fig)
        
        # and another plot!
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for i in range(0, gb_behavior_points.shape[0]) :
            ax.scatter(gb_behavior_points[i,0], gb_behavior_points[i,1], marker="$%d$" % i, alpha=0.3, color='orange')
        
        # all this block of code is just to plot some arrows
        x = gb_behavior_points[:,0]
        y = gb_behavior_points[:,1]
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2)
        ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid", color='orange', alpha=0.3)
        
        ax.set_title("Gradient Boosting predictors (%d predictors)" % len(regressor.estimators_))
        ax.set_xlabel("kPCA dimension 1")
        ax.set_ylabel("kPCA dimension 2")
        
        plt.savefig(os.path.join(task_folder, "kpca_gradient_boosting_predictors_arrows.png"), dpi=300)
        plt.close(fig)        
        
        