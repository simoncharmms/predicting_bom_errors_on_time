""" 
This script implements hierarchical clustering algorithm in order to determine
the optimal number of clusters, based on the previous DBSCAN results.
The next step is the clustering itself, using a non-hierarchical algorithm.
""" 

### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from plots import plot, gca, add_titlebox, fancy_dendrogram
from utils import replacegaps
from constants import FP_DATA

# blockPrint()

start_time = time.time()
next_script = 'benchmark_outlier_detection.py'


def cluster():
    global elapsed_time, fig
    ### ---------------------------------------------------------------------------
    ### Load data.
    ### ---------------------------------------------------------------------------
    # Loading the dataset is quicker using the built-in import.
    akt_ohe_nael_datacsv = pd.read_csv(FP_DATA + 'bom_data.csv', sep=',', encoding='latin-1')
    df = akt_ohe_nael_datacsv
    # bom_data_csv = pd.read_csv('akt_nael_df_sca_rho_ass.csv',sep=',',encoding='latin-1')
    # df = bom_data_csv
    ### ---------------------------------------------------------------------------
    ### Clean and sample data.
    ### ---------------------------------------------------------------------------
    # Column "Unnamed: 0" due to csv.
    del df["Unnamed: 0"]
    # Sample if necessary to partute efficiently.
    # print(df.info())
    # df = df.sample(frac=0.1, replace=True, random_state=1)
    # Clean up and look out for nans.
    df = replacegaps(df)
    # print(df.isna().sum())
    df = df.fillna(0)
    # print(df.isna().sum())
    # print(df.info())
    ### ---------------------------------------------------------------------------
    ### Hierarchical clustering.
    ### ---------------------------------------------------------------------------
    # df = df[df["component"]==5]
    # Get all component.
    subpdln = df["component"].unique()
    subpdln_list = subpdln.tolist()
    no_clusters_list = []
    start_time_loop = time.time()
    for subpdln in subpdln_list:
        df_sub = df[df["component"] == subpdln]
        df_hc = df_sub[["part", "erroneous", "rho_v"]]
        print('Currently clustering: ' + str(subpdln))

        # column = "component"
        # X = "rho_v"
        # # column = "scop"
        # x = df.columns.get_loc(X)
        # y = df.columns.get_loc(column)

        # df_hc = df.iloc[:, [x,y]].values

        ### ---------------------------------------------------------------------------
        ### Hierarchical clustering to determine optimal number of clusters.
        ### ---------------------------------------------------------------------------

        start_time_link = time.time()

        # Generate the linkage matrix.
        Z = linkage(df_hc, 'single', 'euclidean')

        # Check the Cophenetic Correlation Coefficient.
        # c, coph_dists = cophenet(Z, pdist(df_hc))
        # c #The closer the value is to 1,
        # the better the clustering preserves the original distances.

        elapsed_time = time.time() - start_time_link
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        # max_d = maxdists(Z).mean()
        max_c = df_hc['part'].nunique()

        # partute inconsistency based clusters.
        clusters = fcluster(Z, max_c, criterion='maxclust')
        no_clusters = str(max(clusters))
        print("clusters determined by maxclus (count of component): " + no_clusters)

        no_clusters_list.append(no_clusters)

        # calculate full dendrogram
        # plt.figure(figsize=(25, 10))
        # plt.title('Hierarchical Clustering Dendrogram: ' + str(subpdln) +
        #           ' , Cophenetic Correlation Coefficient: %1.2f' % c)
        # plt.xlabel('Sample index $R_v$.')
        # plt.ylabel('Euclidean distance  ' + str(subpdln))
        # dendrogram(
        #     Z,
        #     leaf_rotation=90.,  # rotates the x axis labels
        #     leaf_font_size=8.,  # font size for the x axis labels
        # )
        # #plt.savefig("akt_dendrogram_"+subpdln+".png", dpi=300)
        # plt.close
        # Plot clustered data points.

        # plt.figure(figsize=(10, 8))
        # plt.scatter(df_hc["component"], df_hc["rho_v"], c=clusters, cmap='coolwarm', s=15)
        # plt.title('Hierarchical clustering (Inconsistency): ' + str(subpdln))
        # plt.xlabel('Expected no. of clusters: '+ no_clusters)
        # plt.ylabel(str(subpdln))
        # #plt.savefig("akt_hierarchical_cluster_inc_"+str(subpdln)+".png", dpi=300)
        # plt.close

        # # Plot dendrogram with horizontal max_d marker and truncated annotations.
        # fancy_dendrogram(
        # Z,
        # truncate_mode='lastp',
        # p=12,
        # leaf_rotation=90.,
        # leaf_font_size=12.,
        # show_contracted=True,
        # annotate_above=10,
        # max_d=1.15,
        # )
        # plt.title('Hierarchical clustering dendrogram (truncated): ' + str(subpdln) +
        #   '. partuted c: ' + str(max_c) + ' (l = 1.15).')
        # plt.xlabel('Sample index')
        # plt.ylabel('Euclidean distance  ' + str(subpdln))
        # #plt.savefig("akt_dendrogram_trunc"+str(subpdln)+".png", dpi=300)
        # plt.close

        ### ---------------------------------------------------------------------------
        ### K-means clustering.
        ### ---------------------------------------------------------------------------

        X = df_hc
        y = df_hc["erroneous"]

        # k-means clustering.
        est = KMeans(n_clusters=int(no_clusters))
        est.fit(X)
        labels = est.labels_

        # 3D-plot of clustered data.
        fig = plt.figure(figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.scatter(X["rho_v"], X["erroneous"], X["part"],
                   c=labels.astype(float), cmap="vlag")

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('$R_v$')
        ax.set_ylabel('$erroneous$')
        ax.set_zlabel('part')
        ax.set_title("$K-means$ clustered data of part " + str(subpdln) +
                     ". \n Number clusters determined by adjusted $i$: " +
                     no_clusters + ".")
        ax.dist = 12
        # plt.savefig("akt_clustered_data"+str(subpdln)+".png", dpi=300)
        plt.close
    elapsed_time = time.time() - start_time_loop
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    ### ---------------------------------------------------------------------------
    ### Cluster all date at once by summing up samples.
    ### ---------------------------------------------------------------------------
    df_clus = pd.DataFrame(no_clusters_list, index=subpdln_list)
    clusters = df_clus[0].astype('int32')
    no_clusters = clusters.sum()
    ### ---------------------------------------------------------------------------
    ### K-means clustering.
    ### ---------------------------------------------------------------------------
    X = df[["rho_v", "erroneous", "part"]]
    y = df["erroneous"]
    # k-means clustering.
    est = KMeans(n_clusters=int(no_clusters))
    est.fit(X)
    labels = est.labels_
    # 3D-plot of clustered data.
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X["rho_v"], X["erroneous"], X["part"],
               c=labels.astype(float), cmap="vlag")
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('$R_v$')
    ax.set_ylabel('$erroneous$')
    ax.set_zlabel('$part$')
    ax.set_title("$K-means$ clustered data of all $part$. \n "
                 "Number clusters determined by adjusted $i$: " +
                 str(no_clusters) + ".")
    ax.dist = 12
    # plt.savefig("akt_clustered_data.png", dpi=300)
    plt.close
    ### ---------------------------------------------------------------------------
    ### Store clustered dataset.
    ### ---------------------------------------------------------------------------
    # Append clusters to dataset.
    df["clus"] = labels
    # Clean up and look out for nans.
    df = replacegaps(df)
    # print(df.isna().sum())
    df = df.fillna(0)
    # print(df.isna().sum())
    # print(df.info())
    # Store dataset with cluster labels.
    df.to_csv(FP_DATA + 'bom_data_clus.csv')


### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

# enablePrint

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))