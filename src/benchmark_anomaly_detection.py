""" 
This script implements the Density-Based Spatial Clustering of Applications 
with Noise (DBSCAN) algorithm for anomaly detection.
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
import scipy.sparse as sparse
import sklearn.preprocessing as prep
from sklearn.cluster import DBSCAN


project_path = 'D:/06_Master_Thesis/03_Scripts/Master_thesis.spyproject'
df_path = 'D:/06_Master_Thesis/01_Data'
fig_path = 'D:/06_Master_Thesis/02_Thesis/03_Figures'

os.chdir(project_path)

from plots import plot, gca, add_titlebox
from utils import onehotencoding, replacegaps


# blockPrint()

start_time = time.time()
next_script = 'benchmark_error_prediction.py'

### ---------------------------------------------------------------------------
### Load data.
### ---------------------------------------------------------------------------

os.chdir(df_path)

# Loading the dataset is quicker using the built-in import.
bom_data_csv = pd.read_csv(FP_DATA + 'bom_data.csv',sep=',',encoding='latin-1')
df = bom_data_csv


def dbscan():
    global df, fig
    ### ---------------------------------------------------------------------------
    ### Clean and sample data.
    ### ---------------------------------------------------------------------------
    # Column "Unnamed: 0" due to csv.
    del df["Unnamed: 0"]
    del df["nael"]
    # Sample if necessary to compute efficiently.
    print(df.info())
    # df = df.sample(frac=0.1, replace=True, random_state=1)
    # Clean up and look out for nans.
    df = replacegaps(df)
    print(df.isna().sum())
    df = df.fillna(0)
    print(df.isna().sum())
    print(df.info())
    ### ---------------------------------------------------------------------------
    ### One Hot Encoding (OHE) and Density-Based Spatial Clustering of 
    ### Applications with Noise (DBSCAN).
    ### ---------------------------------------------------------------------------
    # DBSCAN hyperparameters.
    epsilon = 15
    min_samples = 5
    # df = df[["kogr", "sca"]]
    # df = df.sample(frac=0.1, replace=True, random_state=1)
    for column in df:
        try:
            X = "kogr"
            Y = column
            print(column)

            df_dbscan = df[[X, Y]]

            ### ---------------------------------------------------------------------------
            ### Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
            ### ---------------------------------------------------------------------------

            # Compute DBSCAN
            db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(df_dbscan)
            labels = db.labels_

            no_clusters = len(np.unique(labels))
            no_noise = np.sum(np.array(labels) == -1, axis=0)

            print('Estimated no. of clusters: %d' % no_clusters)
            print('Estimated no. of noise points: %d' % no_noise)

            df_dbscan['DBSCAN_opt_labels'] = labels
            df_dbscan['DBSCAN_opt_labels'].value_counts()

            # Plotting the resulting clusters
            os.chdir(fig_path)

            colors = df_dbscan['DBSCAN_opt_labels']

            fig, ax = plt.subplots(figsize=(10, 10))
            plt.scatter(df_dbscan[X], df_dbscan[Y], c=colors, s=15)
            plt.title('DBSCAN ' + str(X) + ', ' + str(Y))
            plt.xlabel('Estimated no. of clusters: %d' % no_clusters)
            plt.ylabel(Y)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            fig.align_labels()
            #plt.savefig("akt_dbscan_" + str(Y) + ".png", dpi=300)

            # Remove noise.
            df_dbscan_clean = df_dbscan[df_dbscan['DBSCAN_opt_labels'] != -1]
            colors = df_dbscan_clean['DBSCAN_opt_labels']

            fig, ax = plt.subplots(figsize=(10, 10))
            plt.scatter(df_dbscan_clean[X], df_dbscan_clean[Y], c=colors, s=15)
            plt.title('DBSCAN ' + str(X) + ', ' + str(Y))
            plt.xlabel('No. of removed noise points: %d' % no_noise)
            plt.ylabel(Y)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            fig.align_labels()
            #plt.savefig("akt_dbscan_clean" + str(Y) + ".png", dpi=300)

        except:
            pass




### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

# enablePrint

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# os.chdir(project_path)
# os.system(next_script)
