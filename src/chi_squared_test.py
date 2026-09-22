""" 
This script implements a Multiple Correspondence Analysis (MCA).
Results will specifiy the dataset for the Procrustes approach.
""" 

### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------

import os 
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN

from constants import FP_DATA
from utils import categorical_segment, continous_segment, replacegaps, ChiSquare
# blockPrint()

start_time = time.time()
next_script = 'benchmark_error_prediction.py'


def chi_squared_test():
    global fit
    ### ---------------------------------------------------------------------------
    ### Load data.
    ### ---------------------------------------------------------------------------
    # Loading the dataset is quicker using the built-in import.
    bom_data_csv = pd.read_csv(FP_DATA + 'bom_data.csv', sep=',', encoding='latin-1')
    df = bom_data_csv
    ### ---------------------------------------------------------------------------
    ### Clean and sample data.
    ### ---------------------------------------------------------------------------
    # Column "Unnamed: 0" due to csv.
    del df["Unnamed: 0"]
    # Sample if necessary to compute efficiently.
    # print(df.info())
    # df = df.sample(frac=0.01, replace=True, random_state=1)
    # Clean up and look out for nans.
    df = replacegaps(df)
    # print(df.isna().sum())
    df = df.fillna(0)
    # print(df.isna().sum())
    print(df.info())
    ### ---------------------------------------------------------------------------
    ### Chi-squared tests.
    ### ---------------------------------------------------------------------------
    # Initialize ChiSquare Class
    cT = ChiSquare(df)
    categorical_var = list(df.dtypes.loc[df.dtypes == 'object'].index)
    print(len(categorical_var))
    print(categorical_var)
    continuous_var = list(df.dtypes.loc[df.dtypes == 'float64'].index)
    print(len(continuous_var))
    print(continuous_var)
    # Feature Selection
    for var in categorical_var:
        cT.TestIndependence(colX=var, colY="erroneous")

    # # A0VA test
    # import scipy.stats as stats
    # for var in continuous_var:
    #     result = stats.f_oneway(df[var][df['erroneous'] == 1],
    #                             df[var][df['erroneous'] == 0])
    #     print(var)
    #     print(result)
    from sklearn.feature_selection import SelectKBest
    from scipy.stats import ttest_ind
    t_stat = []
    for var in continuous_var:
        var_0_erroneous = df[var][df["erroneous"] == "0"]
        var_1_erroneous = df[var][df["erroneous"] == "1"]
        t_value = ttest_ind(var_0_erroneous, var_1_erroneous, equal_var=False)
        print(var)
        print(t_value)
        t_stat.append(t_value)
    # ### ---------------------------------------------------------------------------
    # ### Recursive Factor Elimination.
    # ### ---------------------------------------------------------------------------
    # # first convert all the string columns to categorical form
    # for var in categorical_var:
    #     df[var] = df[var].astype('category')
    # df[categorical_var] = df[categorical_var].apply(lambda x: x.cat.codes)
    # target = df['erroneous']
    # all_columns = list(df.columns)
    # all_columns.remove('erroneous')
    # import warnings
    # warnings.filterwarnings('ignore')
    # from sklearn.feature_selection import RFE
    # from sklearn.linear_model import LogisticRegression
    # X = df[all_columns]  # Features
    # y = df['erroneous']  # Target variable
    # # Feature extraction
    # model = LogisticRegression()
    # rfe = RFE(model, 8)
    # fit = rfe.fit(X, y)
    # print("Num Features: %s" % (fit.n_features_))
    # print("Selected Features: %s" % (fit.support_))
    # print("Feature Ranking: %s" % (fit.ranking_))
    # selected_features_rfe = list(fit.support_)
    # final_features_rfe = []
    # for status, var in zip(selected_features_rfe, all_columns):
    #     if status == True:
    #         final_features_rfe.append(var)
    # final_features_rfe


### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

# enablePrint

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

