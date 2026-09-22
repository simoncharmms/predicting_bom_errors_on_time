""" 
This script implements a Multiple Correspondence Analysis (MCA).
Results will specifiy the dataset for the Procrustes approach.
"""
### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------
#%%
import pandas as pd
import os
import time
import numpy as np
# import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import orthogonal_procrustes

import plots
from plots import plot, gca, add_titlebox
from utils import mapandclean, replacegaps, convert_times
from utils import splitDataFrameList, droprows, blockPrint, enablePrint
from utils import create_adjacency, onehotencoding, procrustes
from constants import FP_DATA, SAMPLE

start_time = time.time()
# blockPrint()
#%%
def solve_orthogonal_procrustes():
    ### ---------------------------------------------------------------------------
    ### Load data.
    ### ---------------------------------------------------------------------------

    # Loading the dataset is quicker using the built-in import.
    bom_data_csv = pd.read_csv(FP_DATA + 'raw_bom_data.csv',sep=',',
                               encoding='latin-1',
                               low_memory=False
                               )
    df = bom_data_csv

    if SAMPLE:
        # Sample data to avoid long runtimes.
        df = df.sample(frac=0.01, replace=True, random_state=1)

    ### ---------------------------------------------------------------------------
    ### Clean and sample data.
    ### ---------------------------------------------------------------------------

    # Column "Unnamed: 0" due to csv.
    del df["Unnamed: 0"]

    # Sample if necessary to compute efficiently.
    # print(df.info())
    # df = df.sample(frac=0.001, replace=True, random_state=1)

    # Clean up and look out for nans.
    df = replacegaps(df)
    # print(df.isna().sum())
    df = df.fillna(0)
    # print(df.isna().sum())
    # print(df.info())

    ### ---------------------------------------------------------------------------
    ### Create adjecency matrix for each unique value of column series.
    ### ---------------------------------------------------------------------------
    '''
    Create the adjacency matrix using one hot encoded components and parts.
    '''

    # Create empty lists for adjacency matrices (ad) and scales.
    grouped_timestamp_list = []
    grouped_component_list = []
    grouped_part_list = []
    grouped_r_list = []
    grouped_sca_list = []
    grouped_rho_v_list = []

    # Loop through timestamp, component, then through part, create adjacency matrices and
    # compute scale and rho_v.
    '''
    This function is algorithm 1 from the paper.
    '''
    # Get all timestamp.
    timestamp = df["timestamp"].astype(str).unique()
    timestamp_list = timestamp.tolist()
    for timestamp in timestamp_list:

        '''
        For bldp-specific plots:
        timestamp = "VS0"
        '''

        timestamp_df = df[df['timestamp'] == int(float(timestamp))]

        print("Currently computing timestamp ", str(timestamp))

        # Get all series.
        component = df["component"].astype(str).unique()
        component_list = component.tolist()

        for component in component_list:
            '''
            For component-specific plots:
            component = "BMW_3_Series"
            '''
            try:
                component_df = timestamp_df[timestamp_df['component'] == int(float(component))]
                component_ad = create_adjacency(component_df)

                part = component_df["part"].astype(str).unique()
                part_list = part.tolist()
                part_no = len(part_list)

                print("Currently computing component ", str(component))

                # Plot the orthogonal adjacency matrix.
                # fig, ax = plt.subplots(figsize=(6, 6))
                # sns.heatmap(component_ad, xticklabels = False, yticklabels = False, cbar=False)
                # title = ax.set_title('Heatmap of the ' + str(component) + ', ' + str(timestamp)
                #                      + ' adjacency matrix. \n'
                #                      'Number of derivatives: ' + str(part_no))
                # #plt.savefig("akt_adjacency_"+str(component)+str(timestamp)+".png", dpi=300)

                for part in part_list:
                    '''
                    Wwe iterate through the dataframe and filter to one part at a time,
                    so that we can partare it to the adjacency matrix of its respective component.
                    Due to one hot-encoding of the real world data, zero-size arrays can occur,
                    which have to be passed.
                    '''
                    part_df = component_df[component_df['part'] == int(float(part))]
                    part_ad = create_adjacency(part_df)

                    # Plot the adjacency matrix.
                    # fig, ax = plt.subplots(figsize=(6, 6))
                    # sns.heatmap(part_ad, xticklabels = False, yticklabels = False, cbar=False)
                    # title = ax.set_title('Heatmap of the ' + str(part) + ', ' +
                    #                      str(timestamp) + ' adjacency matrix.')
                    # #plt.savefig("akt_adjacency_"+str(part)+str(timestamp)+".png", dpi=300)

                    # Get the difference between the component ad and the respective part ad.
                    diff = component_ad.shape[0] - part_ad.shape[0]
                    n = int(diff / 2) + (diff % 2 > 0)  # Round up at .5.
                    # Pad zeros, so that both have ads have the same shape.
                    part_ad = np.pad(part_ad, ((n, n), (n, n)), mode='constant')
                    if diff % 2 > 0:
                        part_ad = np.delete(part_ad, (0), axis=0)
                        part_ad = np.delete(part_ad, (0), axis=1)

                    # compute orthogonal procrustes R and scale.
                    R, sca = orthogonal_procrustes(component_ad, part_ad)

                    # # Plot the orthogonal matrix.
                    # fig, ax = plt.subplots(figsize=(6, 6))
                    # sns.heatmap(part_R, cmap="vlag", xticklabels = False, yticklabels = False, cbar=False)
                    # title = ax.set_title('Heatmap of the ' + str(part) + ' orthogonal matrix.')
                    # #plt.savefig("akt_orthogonal_"+str(part)+str(timestamp)+".png", dpi=300)

                    # compute Rho_V (Escoufier 1973).
                    rho_v = np.sqrt(np.trace(R.T * R))
                    # Append to lists.
                    grouped_part_list.append(timestamp + part)
                    grouped_r_list.append(R)
                    grouped_sca_list.append(sca)
                    grouped_rho_v_list.append(rho_v)
                    # Print results.
                    print(str(part) + ", " + str(timestamp) + " scale is ", str("{:.2f}".format(sca)))
                    print(str(part) + ", " + str(timestamp) + " rho_v is ", str("{:.2f}".format(rho_v)))
            except: pass

    ### ---------------------------------------------------------------------------
    ### Create dataframes.
    ### ---------------------------------------------------------------------------

    # Create dataframes.
    df_sca = pd.DataFrame(grouped_sca_list, index=grouped_part_list)
    df_rho_v = pd.DataFrame(grouped_rho_v_list, index=grouped_part_list)

    # Store as csv.
    df_sca.to_csv(FP_DATA + 'df_sca.csv')
    df_rho_v.to_csv(FP_DATA + 'df_rho.csv')

    df = bom_data_csv
    df = replacegaps(df)
    df = df.fillna(0)

    df_rho_v["mapping"] = df_rho_v.index.astype(str)
    df["mapping"] = df["timestamp"].astype(str) + df["part"].astype(str)
    df = pd.merge(df, df_rho_v, on="mapping", how="left")
    df["rho_v"] = df[0]
    del df["mapping"]
    df = df.reset_index(drop=True)
    # print(df.info())

    # Store dataset with rho_v.
    df.to_csv(FP_DATA + 'bom_data.csv')

    ### ---------------------------------------------------------------------------
    ### Plot selected results.
    ### ---------------------------------------------------------------------------

    # ### ---------------------------------------------------------------------------
    # # Boxplots of sca and rho_v by component.
    # rhov_df = df[["component", "rho_v"]]
    # rhov_df = pd.pivot(rhov_df, columns="component", values="rho_v")
    #
    # fig, ax = plt.subplots(figsize=(40, 4))
    # sns.boxplot(data=rhov_df, color= 'tab:blue')
    # title = ax.set_title('Boxplot of Rv per component.')
    # ax.set_xlabel("component")
    # ax.set_ylabel("Rv")
    #
    # #plt.savefig('akt_rho_v_by_component.png', dpi=300)
    #
    # sca_df = df[["component", "sca"]]
    # sca_df = pd.pivot(sca_df, columns="component", values="sca")
    #
    # fig, ax = plt.subplots(figsize=(40, 4))
    # sns.boxplot(data=sca_df, color= 'tab:blue')
    # title = ax.set_title('Boxplot of SCA per component.')
    # ax.set_xlabel("component")
    # ax.set_ylabel("SCA")
    #
    # #plt.savefig('akt_sca_v_by_component.png', dpi=300)
    #
    #
    # ### ---------------------------------------------------------------------------
    # # Plot Rho_v by hierarchical features.
    # hierarchical_features = ["arct", "sub-pdln", "component", "timestamp"]
    #
    # for x in hierarchical_features:
    #     x_df = df[[x, "rho_v", "erroneous_bool"]]
    #
    #     x_df_erroneous = x_df[x_df["erroneous_bool"] != "No"]
    #     x_df_non_erroneous = x_df[x_df["erroneous_bool"] == "No"]
    #
    #     x_df_erroneous = pd.pivot(x_df_erroneous, columns=x, values="rho_v")
    #     x_df_non_erroneous = pd.pivot(x_df_non_erroneous, columns=x, values="rho_v")
    #
    #     fig, ax = plt.subplots(figsize=(25, 5))
    #     sns.boxplot(data=x_df_erroneous, color= 'tab:red')
    #     title = ax.set_title('Boxplot of $R_v$ per ' + str(x) + ' of $erroneous-parts$.')
    #     ax.set_xlabel(x)
    #     ax.set_ylabel("$R_v$")
    #     #plt.savefig("akt_rho_v_erroneous_by_"+str(x)+".png", dpi=300)
    #
    #     fig, ax = plt.subplots(figsize=(25, 5))
    #     sns.boxplot(data=x_df_non_erroneous, color= 'tab:blue')
    #     title = ax.set_title('Boxplot of $R_v$ per ' + str(x) + ' of non-$erroneous-parts$.')
    #     ax.set_xlabel(x)
    #     ax.set_ylabel("$R_v$")
    #     #plt.savefig("akt_rho_v_non_erroneous_by_"+str(x)+".png", dpi=300)
    #
    # ### ---------------------------------------------------------------------------
    # # Plot $R_v$ values of $erroneous-parts$ and non-$erroneous-parts$ before and after $RLDD$ by $timestamp$.
    # rldd_bool = df[["timestamp", "rldd_bool", "rho_v", "erroneous_bool"]]
    # rldd_bool = pd.DataFrame(rldd_bool)
    # rho_v_mean = round(rldd_bool["rho_v"].mean(),2)
    # erroneous_bool = ["non-", ""]
    #
    # fig, ax = plt.subplots()
    # ax = sns.relplot(
    #     data=rldd_bool, x="timestamp", y="rho_v",
    #     col="erroneous_bool", hue="erroneous_bool", style="rldd_bool",
    #     height=5, aspect=1.75, facet_kws=dict(sharex=False),
    #     kind="line", palette=["tab:blue", "tab:red"]
    # )
    # (ax.map(plt.axhline, y=rho_v_mean, color=".7", dashes=(2, 1), zorder=rho_v_mean)
    #   .set_axis_labels("Mean $R_v$: %0.2f" % rho_v_mean, "$R_v$")
    #   .fig.suptitle("$R_v$ values of $erroneous-parts$ and non-$erroneous-parts$ before and after $RLDD$ by $BLDP_{MP}$.", x=0.5, y=1)
    #   )
    # plt.subplots_adjust(top=0.85)
    # #plt.savefig("akt_rldd_rho_v_by_timestamp.png", dpi=300)
    #
    #
    # # Plot $R_v$ values of $erroneous-parts$ and non-$erroneous-parts$ of the BMW 3 Series F30 and G20 by $timestamp$.
    # part = df[(df["part"] == "F30") | (df["part"] == "G20")]
    # part = part[["timestamp", "part", "rho_v", "erroneous_bool"]]
    # part = pd.DataFrame(part)
    # rho_v_mean = round(part["rho_v"].mean(),2)
    #
    # fig, ax = plt.subplots()
    # ax = sns.relplot(
    #     data=part, x="timestamp", y="rho_v",
    #     col="erroneous_bool", hue="erroneous_bool", style="part",
    #     height=5, aspect=1.75, facet_kws=dict(sharex=False),
    #     kind="line", palette=["tab:blue", "tab:red"]
    # )
    # (ax.map(plt.axhline, y=rho_v_mean, color=".7", dashes=(2, 1), zorder=rho_v_mean)
    #   .set_axis_labels("Mean $R_v$: %0.2f" % rho_v_mean, "$R_v$")
    #   .fig.suptitle("$R_v$ values of $erroneous-parts$ and non-$erroneous-parts$ of the BMW 3 Series F30 and G20 by $BLDP_{MP}$.", x=0.5, y=1)
    #   )
    # plt.subplots_adjust(top=0.85)
    # #plt.savefig("akt_F30_G20_rho_v_by_timestamp.png", dpi=300)
    
### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

#enablePrint()

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# os.chdir(project_path)
# os.system(next_script)