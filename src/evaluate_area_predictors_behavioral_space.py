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
import pickle
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
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
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
    
    if regressor.__class__.__name__.startswith("LGBMRegressor") :
        # LightGBM has a special 'predict' method that makes it possible to get
        # the separate predictions of each predictor in the ensemble
        lgb_predictions = regressor.predict(X, pred_leaf=True).T # transposed, because we need trees as rows and samples as columns
        
        # to compute the semantic points, we now have to subtract y_true from y_pred, for each tree
        semantic_points = np.zeros(lgb_predictions.shape)
        for t in range(0, lgb_predictions.shape[0]) :
            semantic_points[t,:] = np.subtract(y_true, lgb_predictions[t])
    
    elif hasattr(regressor, "estimators_") :
        # this is the standard case for most scikit-learn ensemble regressors;
        # however, they have their annoying little differences
        predictors = regressor.estimators_
        
        # for example, GradientBoosting uses a lists of lists of estimators, with
        # one estimator in each nested list...so, we check if the first element
        # in the list of predictors is a list, and act accordingly
        if isinstance(predictors[0], list) or isinstance(predictors[0], np.ndarray):
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
            
    elif hasattr(regressor, "_predictors") :
        # HistGradientBoosting (that sklearn thinks more or less equivalent to
        # GradientBoosting) has an attribute "_predictors" that should contain
        # all trained decision trees (albeit of a special, internal class);
        # also, it's a list of 1-element lists (...) because in theory you could
        # create several trees at each iteration, but the default is one
        predictors = [ p[0] for p in regressor._predictors ]
        semantic_points = np.zeros((len(predictors), y_true.shape[0]))
        
        # NOTE: **this does not work** as the proprietary predictors require extra
        # arguments, that might maybe be recovered from a trained regressor, but
        # that is more work...
        for i, predictor in enumerate(predictors) :
            semantic_points[i,:] = get_point_in_semantic_space(predictor, X, y_true)
             
    return semantic_points

def create_area_plots(convex_hulls, zero_error_point, colors, folder, figure_title, figure_name) :
    """
    Create plots in behavior space, starting from the information contained
    inside the dictionary of ConvexHull instances
    """
    sns.set_style()
    
    # let's start with one big figure, containing all areas
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    
    # TODO add performance to label
    for regressor_name, [behavior_points, convex_hull] in convex_hulls.items() :
        color = colors[regressor_name]
        patch = Polygon(behavior_points[convex_hull.vertices,:], facecolor=color, edgecolor=color, alpha=0.3)
        ax.scatter(behavior_points[convex_hull.vertices,0], behavior_points[convex_hull.vertices,1], 
                   marker='.', color=color, alpha=0.5, label=regressor_name)
        ax.add_patch(patch)
        
    # also put a reference point, corresponding to no errors in semantic space
    ax.scatter(zero_error_point[0,0], zero_error_point[0,1], marker='x', color='red', label="Reference point, no errors")
    
    ax.legend(loc='best')
    ax.set_xlabel("kPCA dimension 1")
    ax.set_ylabel("kPCA dimensions 2")
    ax.set_title(figure_title)
    
    plt.savefig(os.path.join(folder, figure_name), dpi=300)
    plt.close(fig)
    
    return

def assess_regressor_finished_on_fold(regressor_name, fold_index, folder_task) :
    """
    This function attempts to assess whether a regressor already finished a run
    on a specific fold of a task in the past. In order to understand that, it will
    try to find the files that should have been created.
    """
    is_regressor_finished = True
    
    if not os.path.exists(os.path.join(folder_task, regressor_name + "_behavior_points_fold_%d.csv" % fold_index)) :
        is_regressor_finished = False
        
    if not os.path.exists(os.path.join(folder_task, regressor_name + "_convex_hull_fold_%d.pk" % fold_index)) :
        is_regressor_finished = False
    
    return is_regressor_finished

def assess_all_regressors_finished_on_fold(regressors, fold_index, folder_task) :
    """
    Assess whether all regressors listed in the list 'regressors' have finished
    working on the current fold.
    """
    all_regressors_finished_on_fold = True
    
    for regressor in regressors :
        regressor_name = regressor.__name__
        is_regressor_finished = assess_regressor_finished_on_fold(regressor_name, fold_index, folder_task)
        all_regressors_finished_on_fold = all_regressors_finished_on_fold and is_regressor_finished
        
    return all_regressors_finished_on_fold

def assess_regressor_finished_on_task(regressor_name, n_folds, folder_task) :
    """
    Check for each fold of the task, if the regressor has finished that fold
    """
    is_regressor_finished = True
    
    for fold_index in range(0, n_folds) :
        is_regressor_finished_on_fold = assess_regressor_finished_on_fold(regressor_name, fold_index, folder_task)
        is_regressor_finished = is_regressor_finished and is_regressor_finished_on_fold
        
    return is_regressor_finished

def assess_all_regressors_finished_on_task(regressors, n_folds, folder_task) :
    """
    Check whether all regressors have finished on all folds
    """
    all_regressors_finished = True
    
    for regressor in regressors :
        regressor_name = regressor.__name__
        is_regressor_finished = assess_regressor_finished_on_task(regressor_name, n_folds, folder_task)
        all_regressors_finished = all_regressors_finished and is_regressor_finished
    
    return all_regressors_finished

def create_transformation_semantic_to_behavioral(X, y, random_seed=42, mean=0.0, stdev=0.1, max_depth=3, method="random") :
    """
    This function creates the transformation (a dimensionality reduction) that
    goes from the high-dimensional semantic space (errors on each sample) to
    the 2D behavioral space. We are going to use a KernelPCA with cosine similarity
    as the transformation, because empirically it looks like it works the best for
    our objective. The main difficulty is to obtain enough points in the semantic 
    space to compute a meaningful transformation.
    
    Several methods are possible to obtain points in semantic space:
        - "random": just samples a Gaussian distribution for each dimension
        in the semantic space, a lot of times. Fast, but it completely ignores
        specificities of the dataset (e.g. it would return the same points for
        two datasets with the same sample size, given the same random seed). Even
        if this method looks simple, there are still some hyperparameters, for
        example the mean and the stdev of the Gaussian distribution.
        - "forest": train a large number of decision trees in the same way of
        a random forest, and look at their errors. The issue here is that there
        are extra parameters tied to the trees, for example max depth.
    """
    
    reference_points = None
    
    if method == "random" :
        # easy: create points sampling a standard normal, where the number of
        # columns is the number of samples in the original dataset, and the number
        # of samples is again the number of samples (square matrix)
        reference_points = np_prng.normal(mean, stdev, size=(X.shape[0], X.shape[0]))
        
    elif method == "forest" :
        # train a random forest, imposing a given max depth to its trees
        regressor = RandomForestRegressor(n_estimators=X.shape[0], 
                                          max_depth=max_depth,
                                          n_jobs=max(multiprocessing.cpu_count() - 1, 1), 
                                          random_state=random_seed+10)
        regressor.fit(X, y)
        
        # as each estimator in the random forest is trained independently, we
        # can use their predictions on each sample as reference points
        reference_points = np.zeros((X.shape[0], X.shape[0]))
        
        for i, estimator in enumerate(regressor.estimators_) :
            reference_points[i,:] = estimator.predict(X)
    
    transformation = KernelPCA(n_components=2, kernel='cosine', random_state=random_seed)
    transformation.fit(reference_points)
    
    return transformation
    

if __name__ == "__main__" :
    
    # hard-coded values
    random_seed = 42424242
    folder_main = "evaluate_area_predictors_behavioral_space"
    regressors = [RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor]
    
    performance_metrics = {"R2_train" : r2_score, "R2_test" : r2_score, 
                           "MSE_train" : mean_squared_error, "MSE_test" : mean_squared_error}
    # TODO the code above will not work for the moment, I have to adapt the stuff
    # to a dictionary, from a list
    
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
    for metric_name in [ m for m in performance_metrics.keys() ] + ["area_behavioral_space"] :
        for regressor_class in regressors :
            regressor_name = regressor_class.__name__
            dict_results[regressor_name + "_" + metric_name] = []
    
    print("Loading OpenML metadata...")
    
    # load OpenML-CTR23 benchmark suite
    suite = openml.study.get_suite(353)
    
    # also load a database with all OpenML tasks, just to get some information
    all_tasks = openml.tasks.list_tasks(output_format="dataframe")
    df_tasks_suite = all_tasks.query("tid in @suite.tasks")
    
    # we should probably create all random seeds here, and then fetch them later
    # because if we generate random seeds on the fly, this clashes with skipping
    # datasets or folds for which we already have all the results
    random_seeds = {}
    for task_id in suite.tasks :
        random_seeds[task_id] = []
        task = openml.tasks.get_task(task_id)
        n_repeats, n_folds, n_samples = task.get_split_dimensions()
        
        for i in range(0, n_folds) :
            random_seeds[task_id].append(r_prng.randint(1, 10000))
    
    # iterate over tasks
    for task_index, task_id in enumerate(suite.tasks) :
        print("Now working on task %d, fetching data..." % task_id)
        
        task = openml.tasks.get_task(task_id)
        X_df, y = task.get_X_and_y(dataset_format='dataframe')
        task_dataset_name = df_tasks_suite[df_tasks_suite["tid"] == task_id]["name"].values[0]
        
        print("X.shape =", X_df.shape)
        print("y.shape =", y.shape)
        
        # we have to pre-process the dataset, for example transform
        # categorical variables and the like
        # before that, we also have to take into account missing values,
        # that appear in at least a couple of tasks in this benchmark suite; from
        # a preliminary analysis, it looks like that in each case there are just
        # a few columns with a LOT (52%-100%) of missing values; thus, we can
        # safely remove the columns, running inference in this case does not look appropriate
        X_df = X_df.dropna(axis='columns')
        
        # identify categorical features, replace them with numerical values (integers)
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
        dict_results_fold = {"random_seed" : []}
        for metric_name in [ m for m in performance_metrics.keys() ] + ["area_behavioral_space"] :
            for regressor_class in regressors :
                regressor_name = regressor_class.__name__
                dict_results_fold[regressor_name + "_" + metric_name] = []
                
        # folder where we are going to store all results
        is_task_already_complete = False
        folder_task = os.path.join(folder_main, str(task_id) + "-" + task_dataset_name)
        if not os.path.exists(folder_task) :
            os.makedirs(folder_task)
        else :
            # if the folder exists, we need to check if the file with the fold
            # results also exists; in that case, we read it and we will later
            # use it to check whether some regressors have already been
            # run on the task
            if assess_all_regressors_finished_on_task(regressors, n_folds, folder_task) :
                # set a flag that skips the whole task
                print("For this task, found all results for all regressors, the task will be skipped...")
                is_task_already_complete = True
        
        print("Now going over each fold...")
        if not is_task_already_complete :
            for index_fold in range(0, n_folds) :
                
                print("Working on fold %d..." % index_fold)
                
                # get random seed for this fold
                random_seed_fold = random_seeds[task_id][index_fold]
                
                # get indices for train and test
                train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=index_fold, sample=0)
                
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                
                # computing the kPCA takes quite a lot of time, so we could
                # save some computations by assessing whether ALL regressors have
                # already been processed on this fold for ALL metrics
                # we introduce the check here, because we need to extract the
                # random seed from the pseudo-random number generators
                are_all_regressor_finished_on_fold = assess_all_regressors_finished_on_fold(regressors, index_fold, folder_task)
                
                # if all regressors are finished on this fold, we can just skip;
                # we only need to re-load the dictionary from the CSV
                if are_all_regressor_finished_on_fold :
                    
                    print("All regressors are finished on fold %d, skipping to the next fold..." % index_fold)
                    dict_results_fold = pd.read_csv(os.path.join(folder_task, "%s_folds.csv" % str(task_id))).to_dict(orient='list') # orient here is just to build a structure key -> list
                    print(dict_results_fold)
                
                else :
                    # if not all regressors are done, we just re-compute everything
                    # add random seed to the data
                    dict_results_fold["random_seed"].append(random_seed_fold)
                    
                    # create behavioral space; we have to create it here, because the
                    # shape of the semantic space depends on the number of samples in the
                    # training set, that might change from fold to fold
                    
                    print("Creating behavioral space...")
                    kpca = create_transformation_semantic_to_behavioral(X_train, y_train, method="forest", random_seed=random_seed_fold)
                    
                    # create data structure to store the convex hull instances, so that
                    # later they can be used to plot what is happening in behavior space
                    convex_hulls_fold = {}
                    
                    # normalization should not be necessary for decision-tree based ensembles
                    # so, for the moment let's skip it
                    
                    for regressor_class in regressors :
                        
                        # initialize regressor
                        regressor = initialize_regressor(regressor_class, random_seed=random_seed_fold)
                        regressor_name = regressor_class.__name__
                        
                        # before starting the whole thing, we check if the regressor
                        # has been already applied to this fold; we can evalute this by
                        # assessing whether *ALL* columns that start with the regressor_name
                        # have at least as many elements as the current fold
                        # TODO all this is really cute, but it does not work, because
                        # we also need to recompute the convex hulls...so, we need to first
                        # save all the convex hulls as pickles
                        regressor_data_already_processed = False # TODO change to True when the whole system works
                        for key, values in dict_results_fold.items() :
                            if key.startswith(regressor_name) and len(values) < index_fold + 1 :
                                regressor_data_already_processed = False
                        
                        if not regressor_data_already_processed :
                            print("Training regressor \"%s\" on fold %d..." % (regressor_name, index_fold))
                            regressor.fit(X_train, y_train)
                            
                            y_train_pred = regressor.predict(X_train)
                            y_test_pred = regressor.predict(X_test)
                            
                            # let's go over the metrics, BUT we have to take into account that the name
                            # of the metric also tells us whether we should use training or test set
                            for metric_name, metric in performance_metrics.items() :
                                metric_name_regressor = regressor_name + "_" + metric_name
                                metric_value = 0.0
                                if metric_name.find("test") != -1 :
                                    metric_value = metric(y_test, y_test_pred)
                                else :
                                    metric_value = metric(y_train, y_train_pred)
                                dict_results_fold[metric_name_regressor].append(metric_value)
                            
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
                        
                        else :
                            print("Found all results for regressor \"%s\" on fold %d. Skipping..."
                                  % (regressor_name, index_fold))
                        
                    # save results for the fold
                    df_fold = pd.DataFrame.from_dict(dict_results_fold)
                    df_fold.to_csv(os.path.join(folder_task, "%s_folds.csv" % str(task_id)), index=False)
                    
                    # also save pickles corresponding to the convex hulls AND behavior points
                    # for each regressor in this fold
                    for regressor_name, [behavior_points, hull] in convex_hulls_fold.items() :
                        
                        # save convex hull as pickle
                        with open(os.path.join(folder_task, regressor_name + "_convex_hull_fold_%d.pk" % index_fold), "wb") as fp :
                            pickle.dump(hull, fp)
                            
                        # save behavior points as csv, I guess
                        df_dict_behavior_points = {"kPCA1" : behavior_points[:,0], "kPCA2" : behavior_points[:,1]}
                        df_behavior_points = pd.DataFrame.from_dict(df_dict_behavior_points)
                        df_behavior_points.to_csv(os.path.join(folder_task, regressor_name + "_behavior_points_fold_%d.csv" % index_fold), index=False)
                    
                    # also a few plots in behavior space; but before that, we need
                    # to obtain the behavior point corresponding to "zero errors"
                    zero_error_point = kpca.transform(np.zeros((1, X_train.shape[0])))
                    figure_title = "Behavioral space for %s, fold %d" % (str(task_id) + "-" + task_dataset_name, index_fold)
                    create_area_plots(convex_hulls_fold, zero_error_point, colors, folder_task, figure_title, "areas_behavior_space_fold_%d.png" % index_fold)
        
                    # end if all regressors are finished on fold
                    
                # end for each fold
                
            else :
                # if we end up here, the task is already complete
                print("Task already complete, all files exist. Skipping to the next task...")
                
        # end for each task
        print("Finished working on task")