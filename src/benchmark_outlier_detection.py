""" 
This script implements IQR-based outlier detection for R_v.
""" 

### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------

import os 
import time
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from scipy.stats import norm
from sklearn import ensemble
from sklearn.model_selection import train_test_split


from plots import plot, gca, add_titlebox, fancy_dendrogram
from utils import replacegaps

# blockPrint()

start_time = time.time()
next_script = 'logistic_regression.py'

def benchmark_outlier_detection():
    global mean, std, method, preds
    ### ---------------------------------------------------------------------------
    ### Load data.
    ### ---------------------------------------------------------------------------
    # Loading the dataset is quicker using the built-in import.
    akt_ohe_nael_datacsv = pd.read_csv(FP_DATA+ 'bom_data_clus.csv', sep=',', encoding='latin-1')
    df = akt_ohe_nael_datacsv
    ### ---------------------------------------------------------------------------
    ### Clean and sample data.
    ### ---------------------------------------------------------------------------
    # Column "Unnamed: 0" due to csv.
    del df["Unnamed: 0"]
    # Sample if necessary to compute efficiently.
    # print(df.info())
    # df = df.sample(frac=0.1, replace=True, random_state=1)
    # Clean up and look out for nans.
    df = replacegaps(df)
    # print(df.isna().sum())
    df = df.fillna(0)
    # print(df.isna().sum())
    print(df.info())
    column_names = ["comp", "sers", "lcim", "bldp_mp", "istp_bool",
                    "nlbw", "scop", "rldd_bool", "sca", "rho_v", "conf", "conf", "bdqv_bool"]
    X = df[column_names]
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=0)
    train_df = pd.DataFrame(X_train, columns=column_names)
    test_df = pd.DataFrame(X_test, columns=column_names)
    train_labels = train_df["sers"]
    test_labels = test_df["sers"]
    Rv = df["rho_v"]
    ### ---------------------------------------------------------------------------
    ### Normalize.
    ### ---------------------------------------------------------------------------
    mean = train_df.mean(axis=0)
    std = train_df.std(axis=0)
    train_df = (train_df - mean) / std
    test_df = (test_df - mean) / std
    x_train = train_df["rho_v"]
    x_test = test_df["rho_v"]

    def denorm_Rv(x):
        return x * std["rho_v"] + mean["rho_v"]

    x_train_denorm = denorm_Rv(x_train)
    x_test_denorm = denorm_Rv(x_test)
    # Reformat data for statsmodels.
    X_train = sm.add_constant(x_train)
    X_test = sm.add_constant(x_test)
    ### ---------------------------------------------------------------------------
    ### Initialize result.
    ### ---------------------------------------------------------------------------
    # Dataset per method, quantile, and x value.
    METHODS = ['QuantReg']
    QUANTILES = [0.25, 0.75]
    # QUANTILES.reverse()  # Test out to see if we're getting different results.
    quantiles_legend = [str(int(q * 100)) + 'th percentile' for q in QUANTILES]
    # sns.set_palette(sns.color_palette('Blues', len(QUANTILES)))
    sns.set_palette(sns.color_palette('Blues'))
    # Set dots to a light gray
    dot_color = sns.color_palette('coolwarm', 3)[1]
    preds = np.array([(method, q, x)
                      for method in METHODS
                      for q in QUANTILES
                      for x in x_test])
    preds = pd.DataFrame(preds)
    preds.columns = ['method', 'q', 'x']
    preds = preds.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    preds['label'] = np.resize(test_labels, preds.shape[0])

    ### ---------------------------------------------------------------------------
    ### Quantile loss example.
    ### ---------------------------------------------------------------------------
    # pandas version rather than Keras.
    def quantile_loss(q, y, f):
        # q: Quantile to be evaluated, e.g., 0.5 for median.
        # y: True value.
        # f: Fitted or predicted value.
        e = y - f
        return np.maximum(q * e, (q - 1) * e)

    quantile_loss_example_e = np.linspace(-1, 1, 1000)
    quantile_loss_example_loss_25 = quantile_loss(0.25, 0, quantile_loss_example_e)
    quantile_loss_example_loss_75 = quantile_loss(0.75, 0, quantile_loss_example_e)
    with sns.color_palette('Blues', 3):
        plt.plot(quantile_loss_example_e, quantile_loss_example_loss_25)
        plt.plot(quantile_loss_example_e, quantile_loss_example_loss_75)
        plt.legend([str(int(q * 100)) + 'th percentile' for q in [0.1, 0.5, 0.9]])
        sns.despine(left=True, bottom=True)
        plt.xlabel('Error')
        plt.ylabel('Quantile loss')
        plt.title('Quantile loss by error and quantile', loc='left');
    quantile_loss_example_q = np.linspace(0.01, 0.99, 99)
    quantile_loss_example_loss_neg1 = quantile_loss(quantile_loss_example_q, 0, -1)
    quantile_loss_example_loss_pos1 = quantile_loss(quantile_loss_example_q, 0, 1)
    with sns.color_palette('Blues', 2):
        plt.plot(quantile_loss_example_q, quantile_loss_example_loss_neg1)
        plt.plot(quantile_loss_example_q, quantile_loss_example_loss_pos1)
        plt.legend(['Error of -1', 'Error of +1'])
        sns.despine(left=True, bottom=True)
        plt.xlabel('Quantile')
        plt.ylabel('Quantile loss')
        plt.title('Quantile loss by quantile and error', loc='left');
    ### ---------------------------------------------------------------------------
    ### Scatterplot of Rv.
    ### ---------------------------------------------------------------------------
    Rv.mean()
    Rv.mean()
    from matplotlib.ticker import FuncFormatter
    ax = plt.scatter(x_train_denorm, train_labels, color=dot_color)
    plt.title('Rv vs. value, Boston housing dataset (training slice)', loc='left')
    sns.despine(left=True, bottom=True)
    ax.axes.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: '{:.0%}'.format(x / 100)))
    ax.axes.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: '${:.0f}k'.format(y)))
    plt.xlabel('Proportion of owner-occupied units built prior to 1940')
    plt.ylabel('Median value of owner-occupied homes')
    plt.show()
    ### ---------------------------------------------------------------------------
    ### QuantReg.
    ### ---------------------------------------------------------------------------
    quantreg = sm.QuantReg(train_labels, X_train)  # Don't fit yet, since we'll fit once per quantile.
    preds.loc[preds.method == 'QuantReg', 'pred'] = np.concatenate(
        [quantreg.fit(q=q).predict(X_test) for q in QUANTILES])
    ### ---------------------------------------------------------------------------
    ### Visualize quantiles.
    ### ---------------------------------------------------------------------------
    for i, method in enumerate(METHODS):
        ax = plt.scatter(x_test_denorm, test_labels, color=dot_color)
        plt.plot(preds[preds.method == method].pivot_table(
            index='method', columns='q', values='pred'))
        plt.legend(quantiles_legend)
        # Reversing legend isn't working, possibly because of multiple plots.
        #     handles, labels = ax.get_legend_handles_labels()
        #     ax.legend(handles[::-1], labels[::-1])
        plt.xlim((0, 100))
        ax.axes.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: '{:.0%}'.format(x / 100)))
        ax.axes.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: '${:.0f}k'.format(y)))
        plt.xlabel('Proportion of owner-occupied units built prior to 1940')
        plt.ylabel('Median value of owner-occupied homes')
        plt.title(method + ' quantiles', loc='left')
        sns.despine(left=True, bottom=True)
        plt.show()

### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#os.chdir(project_path)
#os.system(next_script)