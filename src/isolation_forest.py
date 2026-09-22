""" 
This script implements outlier detection based on isolation forest.
""" 

### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------

import os 
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from plots import plot, gca, formatter, add_titlebox, fancy_dendrogram
from utils import replacegaps
from constants import FP_DATA, RANDOM_SEED
# blockPrint()

start_time = time.time()
next_script = 'logistic_regression.py'

### ---------------------------------------------------------------------------
### Parameters.
### ---------------------------------------------------------------------------

n_estimators=1000
max_samples=25019
# contamination was hard-coded to 0.05 against an observed positive rate of
# 0.24 % - a 20x mis-specification that guaranteed ~17k spurious anomalies.
# "auto" lets sklearn use the original Liu et al. offset instead; setting
# IFOREST_CONTAMINATION overrides it with the empirical base rate.
contamination = os.environ.get("IFOREST_CONTAMINATION", "auto")
try:
    contamination = float(contamination)
except ValueError:
    pass
max_features=14  # capped to the real feature count below


def isolation_forest():
    ### ---------------------------------------------------------------------------
    ### Load data.
    ### ---------------------------------------------------------------------------
    # Loading the dataset is quicker using the built-in import.
    akt_ohe_nael_datacsv = pd.read_csv(FP_DATA + 'bom_data_clus.csv', sep=',', encoding='latin-1')
    df = akt_ohe_nael_datacsv
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
    ### partute Isolation forest.
    ### ---------------------------------------------------------------------------
    # The published dataset is anonymized: the original internal column names
    # (lcim, indc, istp_bool, nlbw, scop, rldd_bool, sca, conf) map to
    # feature_1..feature_6. Only use columns that actually exist.
    selected_features = ["component", "part", "timestamp",
                         "feature_1", "feature_2", "feature_3",
                         "feature_4", "feature_5", "feature_6",
                         "rho_v", "clus"]
    selected_features = [c for c in selected_features if c in df.columns]
    metrics_df = df[selected_features].copy()
    # metrics_df=pd.pivot_table(df,values='rho_v',index='kogr',columns='timestamp')
    metrics_df = metrics_df.reset_index(drop=True)
    metrics_df.fillna(0, inplace=True)
    metrics_df
    metrics_df.columns
    # specify the 12 metrics column names to be modelled
    # Never feed the label ("erroneous") into an unsupervised detector.
    to_model_columns = [c for c in metrics_df.columns if c != "erroneous"]
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, \
                          max_features=min(max_features, len(to_model_columns)),
                          random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(metrics_df[to_model_columns])
    pred = clf.predict(metrics_df[to_model_columns])
    metrics_df['anomaly'] = pred
    outliers = metrics_df.loc[metrics_df['anomaly'] == -1]
    outlier_index = list(outliers.index)
    # print(outlier_index)
    # Find the number of anomalies and normal points here points classified -1 are anomalous
    print(metrics_df['anomaly'].value_counts())
    ### ---------------------------------------------------------------------------
    ### Store clean dataset.
    ### ---------------------------------------------------------------------------
    '''
    The following files are being saved with "," seperators, not ";".
    '''
    # Save the new dataset.
    df["anom"] = metrics_df['anomaly']
    # Clean up and look out for nans.
    df = replacegaps(df)
    # print(df.isna().sum())
    df = df.fillna(0)
    # print(df.isna().sum())
    # print(df.info())
    # Save cleaned file.
    df.to_csv(FP_DATA + 'bom_data_clus_anom.csv')
    ### ---------------------------------------------------------------------------
    ### Plot selected results.
    ### ---------------------------------------------------------------------------
    # os.chdir(fig_path)
    # # Plot Anomalies by buildphase and Rv.
    # anomaly = metrics_df[["kogr", "anomaly", "rho_v", "erroneous"]]
    # anomaly = pd.DataFrame(anomaly)
    # anomalous = metrics_df[metrics_df['anomaly']==-1]
    # genuine = metrics_df[metrics_df['anomaly']==1]
    # anomaly_ratio = round(len(anomalous)/(len(anomalous)+len(genuine)),4)
    # rho_v_mean = metrics_df["rho_v"].mean()
    # erroneous = ["non-", ""]
    # style_order=[1, -1]
    # fig, ax = plt.subplots()
    # ax = sns.relplot(
    #     data=anomaly, x="kogr", y="rho_v",
    #     hue="anomaly", style="anomaly", style_order=style_order,
    #     height=5, aspect=1.75, facet_kws=dict(sharex=False),
    #     kind="scatter", palette=["tab:red", "tab:blue"]
    # )
    # (ax.map(plt.axhline, y=rho_v_mean, color=".7", dashes=(2, 1), zorder=rho_v_mean)
    #   .set_axis_labels("Anomaly ratio: %0.2f" % anomaly_ratio, "$R_v$")
    #   .fig.suptitle("$R_v$ values of $BDQV-NRCLs$ and non-$BDQV-NRCLs$ by $s(x, n)$ by $BLDP_{MP}$.", x=0.5, y=1)
    #   )
    # plt.subplots_adjust(top=0.85)
    # #plt.savefig("akt_anomaly_rho_v_by_timestamp.png", dpi=300)
    # ### ---------------------------------------------------------------------------
    # # Corelogram of selected features.
    # selected_features = ["kogr", "component", "timestamp", "cstp",
    #                      "rho_v", "conf", "clus", "erroneous", "anom"]
    # cor_df= df[selected_features]
    # fig, ax = plt.subplots()
    # ax = sns.pairplot(cor_df, kind="scatter", hue="anom",palette=["tab:red", "tab:blue"])
    # # ax.fig.suptitle('Corelogram of features with medium correlation.')
    # #plt.savefig('akt_anom_correlogram.png', dpi=300)


### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#os.chdir(project_path)
#os.system(next_script)