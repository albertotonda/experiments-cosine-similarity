# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:55:46 2023

The idea of this code is to try to assess the area described by the decision trees
in different ensembles, evaluated in behavioral space (kPCA reduction of semantic
space, with a cosine similarity kernel). Of course, this should be assessed over
different folds of different datasets. For this, we are going to use OpenML-CTR23
and probably exploit the folds already offered.

@author: Alberto
"""
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import openml
import os
import pandas as pd
import random
import seaborn as sns
import sys

from inspect import signature, Parameter # this is to set the regressors' parameters
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.decomposition import KernelPCA

# regressors
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor, DMatrix # DMatrix is a class that wraps a data structure

# performance metrics
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_area(predictors_behavior_points) :
    
    # compute convex hull
    
    # evaluate 'volume' of the convex hull (in 2D, it's actually an area)
    
    area = 0.0
    return area

def initialize_regressor(regressor_class, random_seed=42424242) :
    
    print("Now initializing regressor \"%s\"" % regressor_class.__name__)
    
    sig = signature(regressor_class.__init__)
    params = sig.parameters # these are not regular parameters, yet
    # we need to convert them to a dictionary
    params_dict = {}
    for p_name, param in params.items() :
        if params[p_name].default != Parameter.empty :
            params_dict[p_name] = params[p_name].default
    
    # a few parameters that apply to most regressors 
    if 'random_seed' in params :
        params_dict['random_seed'] = random_seed
        
    if 'random_state' in params :
        params_dict['random_state'] = random_seed
        
    if 'n_jobs' in params :
        params_dict['n_jobs'] = max(multiprocessing.cpu_count() - 1, 1) # use maximum available minus one; if it is zero, just use one
    
    if 'n_estimators' in params :
        params_dict['n_estimators'] = 100 # it's already 100 for most ensembles, but we could try some hyperparameter tuning
    
    # instantiate the regressor with the flattened dictionary of parameters
    regressor = regressor_class(**params_dict)
    
    return regressor

def get_point_in_semantic_space(predictor, X, y_true) :
    """
    
    """
    
    y_pred = predictor.predict(X)

    return np.subtract(y_true, y_pred)

def get_points_in_semantic_space_from_regressor(regressor, X, y_true) :
    """
    This function has to adapt to different ways that regressors have to encode
    the predictions of the single predictors/estimators in the ensemble.
    
    Most ensemble regressors/classifiers in scikit-learn have an ".estimators_"
    attribute that gives access to the single estimators (default: decision trees)

    """
    semantic_points = None
    
    if hasattr(regressor, "estimators_") :
        # this is the standard case for most scikit-learn ensemble regressors;
        # however, they have their annoying little differences
        predictors = regressor.estimators_
        
        # for example, GradientBoosting uses a lists of lists of estimators, with
        # one estimator in each nested list...so, we check if the first element
        # in the list of predictors is a list, and act accordingly
        if isinstance(predictors[0], list) :
            predictors = [e[0] for e in regressor.estimators_]
        
        semantic_points = np.zeros((len(predictors), y_true.shape[0]))
        
        for i, predictor in enumerate(predictors) :
            semantic_points[i,:] = get_point_in_semantic_space(predictor, X, y_true)
    
    elif hasattr(regressor, "get_booster") :
        # XGBoost has a completely different way of working; we need to extract
        # its trees from the return of a callable method, however we also need
        # to give them a DMatrix class instance, otherwise it does not work
        xb_trees = [tree for tree in regressor.get_booster()]
        semantic_points = np.zeros((len(xb_trees), y_true.shape[0]))
        X_xgb = DMatrix(X_train)
        for i, tree in enumerate(xb_trees) :
            semantic_points[i] = get_point_in_semantic_space(tree, X_xgb, y_true)
             
    return semantic_points

def create_area_plots(convex_hulls, colors, folder, figure_name) :
    """
    Create plots in behavior space, starting from the information contained
    inside the dictionary of ConvexHull instances
    """
    sns.set_style()
    
    # let's start with one big figure, containing all areas
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    
    for regressor_name, [behavior_points, convex_hull] in convex_hulls.items() :
        color = colors[regressor_name]
        patch = Polygon(behavior_points[convex_hull.vertices,:], facecolor=color, edgecolor=color, alpha=0.3)
        ax.scatter(behavior_points[convex_hull.vertices,0], behavior_points[convex_hull.vertices,1], 
                   marker='.', color=color, alpha=0.5, label=regressor_name)
        ax.add_patch(patch)
        
    # also put a reference point in [0.0, 0.0], corresponding to no errors
    ax.scatter([0.0], [0.0], marker='x', color='red', label="Reference point, no errors")
    
    ax.legend(loc='best')
    ax.set_xlabel("kPCA dimension 1")
    ax.set_ylabel("kPCA dimensions 2")
    
    plt.savefig(os.path.join(folder, figure_name), dpi=300)
    plt.close(fig)
    
    return

if __name__ == "__main__" :
    
    # hard-coded values
    random_seed = 42424242
    folder_main = "evaluate_area_predictors_behavioral_space"
    regressors = [RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor]
    regressors = [RandomForestRegressor, XGBRegressor]
    performance_metrics = [r2_score, mean_squared_error]
    # these are colors that will be used to identify the different regressors
    # in the plots
    colors = {
        'GradientBoostingRegressor' : 'orange',
        'LGBMRegressor' : 'yellow',
        'RandomForestRegressor' : 'green',
        'XGBRegressor' : 'gray',
              }
    
    r_prng = random.Random(random_seed) # pseudo-random number generator
    np_prng = np.random.default_rng(seed=random_seed) # same, but from numpy module
    
    # create folder for the experiments
    if not os.path.exists(folder_main) :
        os.makedirs(folder_main)
    
    # code to filter annoying FutureWarnings
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # data structure to store results
    dict_results = {"dataset_id" : [], "dataset_name" : [], "task_id" : [], "n_features" : [], "n_folds" : []}
    
    # we also add a separate entry for the performance and area in behavioral space of each regressor
    for metric_name in [ m.__name__ for m in performance_metrics ] + ["area_behavioral_space"] :
        for regressor_class in regressors :
            regressor_name = regressor_class.__name__
            dict_results[regressor_name + "_" + metric_name] = []
    
    # load OpenML-CTR23 benchmark suite
    suite = openml.study.get_suite(353)
    
    # iterate over tasks
    for task_index, task_id in enumerate(suite.tasks) :
        print("Now working on task %d, fetching data..." % task_id)
        
        task = openml.tasks.get_task(task_id)
        X_df, y = task.get_X_and_y(dataset_format='dataframe')
        
        print("X.shape =", X_df.shape)
        print("y.shape =", y.shape)
        
        # we have to work on the dataset, for example to take into account
        # categorical variables and the like
        categorical_columns = [ c for c in X_df.columns if c not in X_df._get_numeric_data().columns ]
        print("Categorical columns identified:", categorical_columns)
        
        # for each categorical columns, replace values with integers
        for categorical_column in categorical_columns :

            # get unique values
            unique_values = X_df[categorical_column].unique()
            print("Unique values in column \"%s\": %s" % (categorical_column, str(unique_values)))

            # create dictionary unique_value -> index
            replacement_dict = { unique_values[index] : index for index in range(0, len(unique_values)) }

            # replace values
            X_df[categorical_column].replace(replacement_dict, inplace=True)
        
        # after all this pre-processing, we can finally get the numpy X
        X = X_df.values
        
        print("Now getting splits...")
        n_repeats, n_folds, n_samples = task.get_split_dimensions()
        print("This dataset must be evaluated on %d repeats, with %d folds, and %d samples" % (n_repeats, n_folds, n_samples))
        
        # TODO most tasks are evaluated on just one repeat and 10 folds; but
        # some actually are evaluated on 10 repeats! I think I don't care, and
        # just evaluate tasks on 1 repeat and 10 folds
        
        # data structure to record performance and other stuff on the folds
        dict_results_fold = {}
        for metric_name in [ m.__name__ for m in performance_metrics ] + ["area_behavioral_space"] :
            for regressor_class in regressors :
                regressor_name = regressor_class.__name__
                dict_results_fold[regressor_name + "_" + metric_name] = []
                
        # folder where we are going to store all results
        folder_task = os.path.join(folder_main, str(task_id))
        if not os.path.exists(folder_task) :
            os.makedirs(folder_task)
        
        print("Now going over each fold...")
        for index_fold in range(0, n_folds) :
            
            # a random seed for this fold
            random_seed_fold = r_prng.randint(1, 10000)
            
            # get indices for train and test
            train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=index_fold, sample=0)
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            # create behavioral space; we have to create it here, because the
            # shape of the semantic space depends on the number of samples in the
            # training set, that might change from fold to fold
            print("Creating behavioral space...")
            reference_points = np_prng.standard_normal(size=(X_train.shape[0], X_train.shape[0]))
            kpca = KernelPCA(n_components=2, kernel='cosine', random_state=random_seed)
            kpca.fit(reference_points)
            
            # create data structure to store the convex hull instances, so that
            # later they can be used to plot what is happening in behavior space
            convex_hulls_fold = {}
            
            # normalization should not be necessary for decision-tree based ensembles
            # so, for the moment let's skip it
            for regressor_class in regressors :
                
                # initialize regressor
                regressor = initialize_regressor(regressor_class, random_seed=random_seed_fold)
                regressor_name = regressor_class.__name__
                
                print("Training regressor \"%s\" on fold %d..." % (regressor_name, index_fold))
                regressor.fit(X_train, y_train)
                
                y_train_pred = regressor.predict(X_train)
                y_test_pred = regressor.predict(X_test)
                
                for metric in performance_metrics :
                    metric_name = regressor_name + "_" + metric.__name__
                    metric_value = metric(y_test, y_test_pred)
                    dict_results_fold[metric_name].append(metric_value)
                
                # compute semantic points, translate to behavior points, compute area, maybe a plot
                print("Obtaining points in behavior space...")
                semantic_points = get_points_in_semantic_space_from_regressor(regressor, X_train, y_train)
                behavior_points = kpca.transform(semantic_points)
                
                # now that we have the points in behavior space, we can compute
                # their convex hull and obtain lots of interesting information
                # NOTE: 'volume' for a 2D convex hull corresponds to an area
                hull = ConvexHull(behavior_points)
                convex_hulls_fold[regressor_name] = [behavior_points, hull]
                
                dict_results_fold[regressor_name + "_area_behavioral_space"].append(hull.volume)
                
            # save results for the fold
            df_fold = pd.DataFrame.from_dict(dict_results_fold)
            df_fold.to_csv(os.path.join(folder_task, "%s_folds.csv" % str(task_id)), index=False)
            
            # also a few plots in behavior space
            create_area_plots(convex_hulls_fold, colors, folder_task, "areas_behavior_space_fold_%d.png" % index_fold)
            