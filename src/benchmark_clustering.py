""" 
This script implements a spectral clustering algorithm in order to determine
the optimal number of clusters, based on the previous DBSCAN results.
The next step is the clustering itself, using a non-hierarchical algorithm.
""" 
### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------
import os 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
#%matplotlib inline
from sklearn.cluster import SpectralClustering
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from plots import plot, gca, add_titlebox
from utils import onehotencoding, replacegaps
from constants import FP_DATA

# blockPrint()

start_time = time.time()
next_script = 'benchmark_anomaly_detection.py'


def spectral_clustering():
    global columns
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
    # del df["bdqv_clas"]
    # Sample if necessary to compute efficiently.
    print(df.info())
    df = df.sample(frac=0.01, replace=True, random_state=1)
    # Clean up and look out for nans.
    df = replacegaps(df)
    print(df.isna().sum())
    df = df.fillna(0)
    print(df.isna().sum())
    print(df.info())
    columns = df.columns
    ### ---------------------------------------------------------------------------
    ### Spectral clustering. 
    ### ---------------------------------------------------------------------------
    for column in df:
        try:
            X = "sca"
            column = "bldp_mp"
            x = df.columns.get_loc(X)
            y = df.columns.get_loc(column)

            print('Currently clustering: %d' % y + ' (' + column + ')')

            df_sc = df.iloc[:, [x, y]].values

            # Determine clusters from "akt_hierarchical_clustering".
            cluster_dict = {"sers": 21,
                            "scop": 13,
                            "lcim": 2,
                            "bldp_mp": 11,
                            "indc": 4,
                            "istp_bool": 2,
                            "nlbw": 2,
                            "bdqv_bool": 2,
                            "sca": 12
                            }

            n_clusters = cluster_dict[column]

            sc = SpectralClustering(n_clusters=n_clusters).fit(df_sc)
            print(sc)

            labels = sc.labels_

            # Scatter plit.
            plt.figure(figsize=(25, 10))
            plt.title('Spectral Clustering Scatterplot: ' + column)
            plt.xlabel(X)
            plt.ylabel(column)
            plt.scatter(df_sc[:, 0], df_sc[:, 1], c=labels)
            #plt.savefig("akt_spectral_" + column + ".png", dpi=300)
            plt.close

            # Plot clustered data points.
            plt.figure(figsize=(10, 8))
            plt.scatter(df_sc[:, 0], df_sc[:, 1], c=sc.labels_, cmap='coolwarm', s=15)
            plt.title('Spectral clustering (Inconsistency): ' + column)
            plt.xlabel('Expected no. of clusters: ' + str(n_clusters))
            plt.ylabel(column)
            #plt.savefig("akt_spectral_cluster_inc_" + column + ".png", dpi=300)
            plt.close

        except:
            pass
    from sklearn.cluster import AffinityPropagation
    clustering = AffinityPropagation().fit(df_sc)
    clustering
    AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True, damping=0.5, max_iter=200,
                        preference=None, verbose=False)
    # labels
    print(clustering.labels_)
    # cluster centers
    print(clustering.cluster_centers_)
### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------
# enablePrint
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))



