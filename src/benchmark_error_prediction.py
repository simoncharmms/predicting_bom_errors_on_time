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

from sklearn.model_selection import train_test_split 
import keras 
from keras.models import Sequential 
from keras.layers import InputLayer 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.constraints import maxnorm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from plots import plot, gca, add_titlebox
from utils import onehotencoding, replacegaps

# blockPrint()

start_time = time.time()
next_script = '...'

### ---------------------------------------------------------------------------
### Load data.
### ---------------------------------------------------------------------------


# Loading the dataset is quicker using the built-in import.
bom_data_csv = pd.read_csv(FP_DATA + 'bom_data.csv',sep=',',encoding='latin-1')

df = bom_data_csv

### ---------------------------------------------------------------------------
### Clean and sample data.
### ---------------------------------------------------------------------------

# Column "Unnamed: 0" due to csv.
del df["Unnamed: 0"]
#del df["bdqv_clas"]

# Sample if necessary to compute efficiently.
print(df.info())
# df = df.sample(frac=0.1, replace=True, random_state=1)

# Clean up and look out for nans.
df = replacegaps(df)
print(df.isna().sum())
df = df.fillna(0)
print(df.isna().sum())
print(df.info())

selected_features = ["lcim", "derv", "bldp_mp", "kogr", "istp_bool", 
              "nlbw", "conf", "dtcs", "comp", "scop", "sca", "bdqv_bool"]
df = df[selected_features]

columns = df.columns

### ---------------------------------------------------------------------------
### Feature scaling.
### ---------------------------------------------------------------------------

# scaler = PowerTransformer().fit_transform
# df[['sers','lcim','bldp_mp', 'nlbw', 'dtcs', 'comp', 'scop', 
#     'sca']] = scaler(df[['sers','lcim','bldp_mp', 'nlbw', 'dtcs', 
#                          'comp', 'scop', 'sca']])
# df.head(10)

### ---------------------------------------------------------------------------
### Random forest..
### ---------------------------------------------------------------------------

data = df.drop(["bdqv_bool"], axis=1)
data = pd.get_dummies(data)

X = data.astype(int)
y = df.bdqv_bool

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,
                                                    random_state= 42)


rf = RandomForestClassifier(n_estimators=1500, max_depth=500,
                              random_state=42)
rf.fit(X_train, y_train) 
score = rf.score(X_train, y_train)
score2 = rf.score(X_test, y_test)
print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))

rf_predictions = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)

y_pred = rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

fi = pd.DataFrame({'feature': list(X_train.columns),
                   'importance': rf.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi.head()

### ---------------------------------------------------------------------------
### Neural network.
### ---------------------------------------------------------------------------

model = Sequential()
model.add(Dense(64, input_dim=11, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(rate=0.2))
model.add(Dense(8, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='sigmoid')) # relu, hidden schicht groß machen, batch hoch
# Grüne Mittlungskurve plotten

model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)

os.chdir(fig_path)
plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 
plt.title('model accuracy') 
plt.ylabel('accuracy')
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='upper left') 
#FP_DATA + 'bom_data.csv("akt_rnn_accuracy.png", dpi=300)

### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

# enablePrint

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# os.chdir(project_path)
# os.system(next_script)