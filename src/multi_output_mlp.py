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
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import tensorflow as tf
#import pydot
#import graphviz
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support

from plots import plot, gca, formatter, add_titlebox, fancy_dendrogram
from utils import replacegaps
from constants import FP_DATA
# blockPrint()

start_time = time.time()
next_script = 'logistic_regression.py'

def multi_output_mlp():
    ### ---------------------------------------------------------------------------
    ### Load data.
    ### ---------------------------------------------------------------------------

    # Loading the dataset is quicker using the built-in import.
    akt_ohe_nael_datacsv = pd.read_csv(FP_DATA + 'bom_data_clus_anom.csv', sep=',', encoding='latin-1')
    df = akt_ohe_nael_datacsv

    ### ---------------------------------------------------------------------------
    ### Clean and sample data.
    ### ---------------------------------------------------------------------------
    # Column "Unnamed: 0" due to csv.
    del df["Unnamed: 0"]
    # Sample if necessary to partute efficiently.
    print(df.info())
    # df = df.sample(frac=0.1, replace=True, random_state=1)
    # Clean up and look out for nans.
    df = replacegaps(df)
    # print(df.isna().sum())
    df = df.fillna(0)
    # print(df.isna().sum())
    # print(df.info())
    ### ---------------------------------------------------------------------------
    ### Define hyperparameters.
    ### ---------------------------------------------------------------------------
    # Neural Network hyperparameters.
    epochs = 510
    batch_size = 64
    test_size = 0.15
    ### ---------------------------------------------------------------------------
    ### Multilayer perceptron model.
    ### ---------------------------------------------------------------------------
    # Define the target variable and features
    target_reg = 'timestamp'
    target_clas = 'erroneous'
    # Drop targets.
    # features = [x for x in list(df.columns) if x not in [target_reg, target_clas]]
    features = ['component', 'part', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'rho_v', 'clus', 'anom']
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    # from tensorflow.keras.utils import plot_model
    from tensorflow.keras.metrics import Recall
    # num_df = (df.drop(columns, axis=1).join(df[columns].apply(pd.to_numeric, errors='coerce')))
    X = df[features]
    y_reg = df[target_reg].astype(int)
    y_clas = df[target_clas].astype(int)
    # Print the input dataset to LaTeX.
    X_info = round(X.describe())
    # print(X_info.to_latex(index=False))
    n_features = len(features)
    # encode strings to integer
    n_class = 2
    # split data into train and test sets
    X_train, X_test, y_train_reg, y_test_reg, y_train_clas, y_test_clas = train_test_split(X, y_reg, y_clas,
                                                                                           test_size=test_size,
                                                                                           random_state=1)
    # input
    visible = Input(shape=(n_features,))
    hidden1 = Dense(n_features, activation='relu', kernel_initializer='he_normal')(visible)
    hidden2 = Dense(10, activation='relu', kernel_initializer='he_normal')(hidden1)
    # regression output
    out_reg = Dense(1, activation='linear')(hidden2)
    # classification output
    out_clas = Dense(n_class, activation='sigmoid')(hidden2)
    # define model
    model = Model(inputs=visible, outputs=[out_reg, out_clas])
    # partile the keras model
    model.partile(loss=['mse', 'sparse_categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
    # plot graph of model
    # plot_model(model, to_file="akt_mlp_model_"+str(epochs)+"_"+str(batch_size)+".png", show_shapes=True)
    # fit the keras model on the dataset
    model.fit(X_train, [y_train_reg, y_train_clas], epochs=epochs, batch_size=batch_size, verbose=1)
    # model.fit(X_train, y_train_clas, epochs=epochs, batch_size=batch_size, verbose=1)
    # make predictions on test set
    yhat1_test, yhat2_test = model.predict(X_test)
    # calculate error for regression model
    error_test = mean_absolute_error(y_test_reg, yhat1_test)
    print('MAE test: %.3f' % error_test)
    # evaluate accuracy for classification model
    yhat2_test = np.around(yhat2_test[:, 1])
    acc = accuracy_score(y_test_clas, yhat2_test)
    print('Accuracy test: %.3f' % acc)
    # Create confusion matrix for the test dataset.
    conf_matrix = metrics.confusion_matrix(y_test_clas, yhat2_test)
    print(conf_matrix)
    # print(cnf_matrix.to_latex(index=False))
    # make predictions on train set
    yhat1_train, yhat2_train = model.predict(X_train)
    # calculate error for regression model
    error_train = mean_absolute_error(y_train_reg, yhat1_train)
    print('MAE train: %.3f' % error_train)
    # evaluate accuracy for classification model
    yhat2_train = np.around(yhat2_train[:, 1])
    acc = accuracy_score(y_train_clas, yhat2_train)
    print('Accuracy train: %.3f' % acc)
    # print('Accuracy for test set: %0.4f' % accuracy_score(y_test_class, yhat2_test))
    # print('Accuracy for train set: %0.4f' % accuracy_score(y_train_class, yhat2_train))
    # print('\n')
    # print('Precision for test set: %0.4f' % precision_score(y_test_class, yhat2_test))
    # print('Precision for train set: %0.4f' % precision_score(y_train_class, yhat2_train))
    # print('\n')
    # print('Recall for test set: %0.4f' % recall_score(y_test_class, yhat2_test))
    # print('Recall for train set: %0.4f' % recall_score(y_train_class, yhat2_train))
    # train_fpr, train_tpr, train_thresholds = roc_curve(y_train_class, yhat2_train)
    # test_fpr, test_tpr, test_thresholds = roc_curve(y_test_class, yhat2_test)
    # train_roc_auc = auc(train_fpr, train_tpr)
    # test_roc_auc = auc(test_fpr, test_tpr)
    # print('AUC for test set: %0.4f' % test_roc_auc)
    # print('AUC for train set: %0.4f' % train_roc_auc)
    ### ---------------------------------------------------------------------------
    ### Store the model.
    ### ---------------------------------------------------------------------------
    # os.chdir(project_path)
    # from sklearn.externals import joblib
    # # Save to file.
    # mlp_file = "akt_mlp_model_"+str(epochs)+"_"+str(batch_size)+".pkl"
    # joblib.dump(model, mlp_file)
    # # Load from file
    # mlp_model = joblib.load(mlp_file)
    ### ---------------------------------------------------------------------------
    ### Apply to whole dataset and store.
    ### ---------------------------------------------------------------------------
    # # Load new clean dataset.
    # os.chdir(new_df_path)
    # akt_df_new = pd.read_csv("akt_new_nael_data_clean.csv")
    # df = akt_df_new
    # df['cons'] = 0
    # df['conf'] = 0
    # df['cstp'] = 0
    # Create the required feature setting, where possible.
    df_pred = df[features]
    y_reg = df[target_reg]
    y_clas = df[target_clas]
    # Predict.
    yhat1, yhat2 = model.predict(df_pred)
    # calculate error for regression model
    error_test = mean_absolute_error(y_reg, yhat1)
    print('MAE whole dataset: %.3f' % error_test)
    # evaluate accuracy for classification model
    yhat2 = np.around(yhat2[:, 1])
    acc = accuracy_score(y_clas, yhat2)
    print('Accuracy whole dataset: %.6f' % acc)
    # Map binary BDQV prediction to mlpp prediction probabiliy.
    df_pred["mlpp_reg"] = yhat1
    df_pred["mlpp_clas"] = yhat2
    y_true = df["erroneous"]
    df_pred["erroneous"] = df["erroneous"]
    y_true = df_pred["erroneous"]
    y_pred = df_pred["mlpp_clas"]
    # Mark false positives and false negatives.
    df_pred["conf"] = np.where(
        y_true > y_pred, "false negative",
        np.where(y_true < y_pred, "false positive", "correct prediction"))
    df_pred['conf'].value_counts()
    # Create confusion matrix.
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    precision = precision_score(y_true, y_pred)
    print('Precision: %.6f' % precision)
    recall = recall_score(y_true, y_pred)
    print('Recall: %.6f' % recall)
    # Map KOGR and NRCLs back.
    # df_pred["kogr"] = akt_ohe_nael_datacsv["kogr"]
    # df_pred["nael"] = akt_ohe_nael_datacsv["nael"]
    # target_column = df["nael"]
    # keys = list(akt_nrcl_map_ohe['nael'])
    # values = list(akt_nrcl_map_ohe['nael'])
    # map_values = dict(zip(keys, values))
    # mapper = target_column.isin(map_values)
    # df.loc[mapper, 'nael'] = df.loc[mapper, 'nael'].apply(lambda row: map_values[row])
    # # df.fillna(0, inplace=True)
    # df["nael"].unique()
    # # Clean up and look out for nans.
    # df_pred = replacegaps(df_pred)
    # # print(df.isna().sum())
    # df_pred = df_pred.fillna(0)
    # # print(df.isna().sum())
    # # print(df.info())
    # Save cleaned file.
    # os.chdir(df_path)
    df_pred.to_csv(FP_DATA + "mlp_"+str(epochs)+"_"+str(batch_size)+"pred.csv")
    # df_pred = pd.read_csv("akt_mlp_"+str(epochs)+"_"+str(batch_size)+"pred.csv",sep=',',encoding='latin-1')
    ### ---------------------------------------------------------------------------
    ### Plot selected results.
    ### ---------------------------------------------------------------------------
    # # Visualize ROC curve
    # fig, ax = plt.subplots(figsize=(8, 4))
    # plt.plot(test_fpr, test_tpr, color='tab:red', label='ROC curve for test set (area = %0.2f)' % test_roc_auc)
    # plt.plot(train_fpr, train_tpr, color='tab:blue', label='ROC curve for train set (area = %0.2f)' % train_roc_auc)
    # plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.legend(loc="lower right")
    # title = ax.set_title('Count of $NRCLs$ by mapped $BLDP$.')
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # fig.tight_layout()
    # #plt.savefig("akt_mlp_roc_"+str(epochs)+"_"+str(batch_size)+".png", dpi=300)
    ### ---------------------------------------------------------------------------
    # df_pred["dtcs"] = akt_ohe_nael_datacsv["dtcs"]
    # # Visualize predictions
    # selected_features = ["kogr", "feature_1", "feature_4", "dtcs", "rho_v", "clus",
    #                      "anom", "mlpp_reg", "mlpp_clas", "conf"]
    # cor_df= df_pred[selected_features]
    # fig, ax = plt.subplots()
    # ax = sns.pairplot(cor_df, kind="scatter", hue="conf",palette=["tab:blue", "tab:red", "gold"])
    # # ax.fig.suptitle('Corelogram of features with medium correlation.')
    # #plt.savefig("akt_mlp_result_"+str(epochs)+"_"+str(batch_size)+".png", dpi=300)


### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#os.chdir(project_path)
#os.system(next_script)