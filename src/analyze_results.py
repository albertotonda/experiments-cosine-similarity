# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:06:36 2023

Simple script to analyze the results of an experimental run.

@author: Alberto
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

from scipy.stats import ks_2samp

def main() :
    
    # some hard-coded stuff that will be used later
    # root directory with all the experiments in sub-folders
    root_directory = "../results/2023-11-10-RF-space-evaluate_area_predictors_behavioral_space"
    # threshold for the statistical experiments
    p_value_threshold = 0.05
    # set seaborn style (prettier plots)
    sns.set_style('darkgrid')
    
    # get a list of all the subdirectories
    subdirectories = [os.path.join(root_directory, f) for f in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, f))]
    print("Found a total of %d subdirectories, starting analysis..." % len(subdirectories))
    
    list_of_dfs = []
    best_regressors = {}
    best_tied_regressors = {}
    
    smallest_areas = {}
    smallest_tied_areas = {}
    
    largest_areas = {}
    largest_tied_areas = {}
    
    for subdirectory in subdirectories :
        
        # get name of the data set
        directory_basename = os.path.basename(subdirectory)
        dataset_id = directory_basename.split("-")[0]
        dataset_name = directory_basename.split("-")[1]
        print("Working on results for dataset \"%s\", id=%s..." % (dataset_name, dataset_id))
        
        # read the file with the results
        df_experiment = pd.read_csv(os.path.join(subdirectory, dataset_id + "_folds.csv"))
        
        # get all columns that end in "_R2_test" and the ones related to the area in behavioral space
        columns_r2_test = [c for c in df_experiment.columns if c.endswith("_R2_test")]
        regressor_names = [c.split("_")[0] for c in columns_r2_test]
        
        # some computation here, to get means and stdevs of performance
        means = [np.mean(df_experiment[c].values) for c in columns_r2_test]
        stdevs = [np.std(df_experiment[c].values) for c in columns_r2_test]
        
        # a curiosity: can we use Kolmogorov-Smirnov to test if the regressor
        # with the highest average R2 (test) in the experiment is the best also
        # in a stastically significant way?
        sorted_regressors = sorted(list(zip(regressor_names, means)), reverse=True, key=lambda x : x[1])
        print("For this experiment, the most performing regressor is %s, mean R2=%.4f" % (sorted_regressors[0][0], sorted_regressors[0][1]))
        
        column_best_regressor = sorted_regressors[0][0] + "_R2_test"
        is_the_best_significant = True
        equally_performing_regressors = []
        
        for i in range(1, len(sorted_regressors)) :
            column_other_regressor = sorted_regressors[i][0] + "_R2_test"
            stat, p_value = ks_2samp(df_experiment[column_best_regressor], df_experiment[column_other_regressor])
            if p_value > p_value_threshold :
                is_the_best_significant = False
                equally_performing_regressors.append(sorted_regressors[i][0])
                
        if is_the_best_significant :
            print("The performance of the best regressor is statistically significant!")
            if sorted_regressors[0][0] not in best_regressors :
                best_regressors[sorted_regressors[0][0]] = 0
            best_regressors[sorted_regressors[0][0]] += 1
        else :
            print("But other regressors have unseparable performances...")
            for regressor in [sorted_regressors[0][0]] + equally_performing_regressors :
                if regressor not in best_tied_regressors :
                    best_tied_regressors[regressor] = 0
                best_tied_regressors[regressor] += 1
            
        # plot time! what about a nice figure with a barplot of the average performance in test?
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        ax.bar(x=range(0, len(regressor_names)), height=means, yerr=stdevs)
        ax.set_xticks(range(0, len(regressor_names)), regressor_names)
        ax.set_ylabel("R2 (test)")
        ax.set_title("Mean R2 value in test, for data set %s" % dataset_name)
        plt.savefig(os.path.join(root_directory, dataset_id + "-" + dataset_name + "-r2.png"), dpi=300)
        plt.close(fig)
        
        # now, take a look at the area of each ensemble in the behavioral space
        columns_area_behavioral = [c for c in df_experiment.columns if c.endswith("_area_behavioral_space")]
        # some computation here, to get means and stdevs of performance
        means = [np.mean(df_experiment[c].values) for c in columns_area_behavioral]
        stdevs = [np.std(df_experiment[c].values) for c in columns_area_behavioral]
        
        sorted_regressors = sorted(list(zip(regressor_names, means)), key=lambda x : x[1])
        print("For this experiment, the regressor with the smallest area in behavioral space is %s, mean A=%.4e" % 
              (sorted_regressors[0][0], sorted_regressors[0][1]))
        print("For this experiment, the regressor with the largest area in behavioral space is %s, mean A=%.4e" % 
              (sorted_regressors[-1][0], sorted_regressors[-1][1]))
        
        column_best_regressor = sorted_regressors[0][0] + "_area_behavioral_space"
        is_the_best_significant = True
        equally_performing_regressors = []
        # TODO finish this
        
        # add column, with just the name of the dataset
        df_experiment["dataset_name"] = [dataset_name] * df_experiment.shape[0]
        
        # add dataset to the list
        list_of_dfs.append(df_experiment)
        
    # finally, merge everything
    df_global = pd.concat(list_of_dfs)
    print(df_global)
    
    # now that we have a global dataset with all the information, we can have
    # a lot of fun with it! For example, get the global average performance 
    
    # also, some stats of the regressors that are more often the best
    for key, value in best_regressors.items() :
        print("Regressor \"%s\" is the best (statistically significant) for %d/%d datasets" %
              (key, value, len(subdirectories)))        
    
    for key, value in best_tied_regressors.items() :
        print("Regressor \"%s\" is the best (tied with others) for %d/%d datasets" %
              (key, value, len(subdirectories))) 
    
    return

if __name__ == "__main__" :
    sys.exit(main())