import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import keras as ks
import sklearn as skl
import tensorflow as tf
from sklearn.linear_model import  LinearRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold
from sklearn.linear_model import  LogisticRegression as LogR
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.metrics import RootMeanSquaredError
from keras.metrics import MeanSquaredError
from keras.layers import Dense


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)

pd.set_option('display.width', 750)

le = skl.preprocessing.LabelEncoder()

df = pd.read_csv('weather_and_fire.csv')
df = df.dropna()
stuff = ["ETo (in)", "Precip (in)",  "Sol Rad (Ly/day)" ,"Avg Vap Pres (mBars)", "Max Air Temp (F)" ,  "Min Air Temp (F)" ,"Avg Air Temp (F)" ,"Max Rel Hum (%)","Min Rel Hum (%)","Avg Rel Hum (%)",  "Dew Point (F)", "Avg Wind Speed (mph)","Wind Run (miles)", "Avg Soil Temp (F)"]
df = df.astype({stuff[0]: 'float32'})
# print(df)

X = df[stuff]
# df["FIRE_SIZE_CLASS"] = le.fit_transform(df["FIRE_SIZE_CLASS"])
# y = LabelBinarizer().fit_transform(df["FIRE_SIZE_CLASS"])

y = df["FIRE_SIZE"]

# print(y)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def root_mean_squared_percent_error(y_true, y_pred):
    return (np.sqrt(np.mean(np.square((y_true - y_pred)/y_true)))) * 100

def ann():
    model = Sequential()
    model.add(Dense(64, input_dim=14,kernel_initializer="normal", activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(16, kernel_initializer='normal', activation='tanh'))
    model.compile(loss=root_mean_squared_percent_error, optimizer='adam', metrics=['mse','mae',''])
    return model

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.33)

estimator = KerasRegressor(build_fn=ann, epochs=25, batch_size=1, verbose=1)
kfold = KFold(n_splits = 2)
results = cross_val_score(estimator, X_test,Y_test, cv=kfold )
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
estimator.fit(X_train,Y_train)
prediction = estimator.predict(X_test)
print(accuracy_score(Y_test, prediction))



