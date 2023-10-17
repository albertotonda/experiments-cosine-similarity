# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:47:32 2023

Quick script to analyze the OpenML-CTR23 regression dataset.

@author: Alberto
"""

import openml
import pandas as pd

if __name__ == "__main__" :
    
    # load OpenML-CTR23 benchmark suite
    suite = openml.study.get_suite(353)
    
    # prepare data structure that will be later converted to a CSV
    df_dictionary = {
        "dataset_id" : [], "dataset_name" : [],
        "n_samples" : [], "n_features" : [],
        "categorical_features" : []
                     }
    
    tasks = openml.tasks.list_tasks(output_format="dataframe")
    tasks_suite = tasks.query("tid in @suite.tasks")
    
    tasks_suite.to_csv("OpenML-CTR23_overview.csv", index=False)