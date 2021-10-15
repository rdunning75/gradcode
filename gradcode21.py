import pandas as pd
import numpy as np
from keras.initializers.initializers_v2 import RandomNormal
from numpy import mean
from numpy import std
import keras as ks
import sklearn as skl
import tensorflow as tf
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.backend import sqrt, mean, square
from keras.layers import Dense, Dropout


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)
# tf.enable_eager_execution()
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
     rmspe = (sqrt(mean(square((y_true - y_pred) / y_true)))) * 100
     return rmspe

    #   rmspe = (ks.backend.sqrt(ks.backend.mean(ks.backend.square((y_true - y_pred) / y_true)))) * 100
    # print(type(rmspe))
    # thing = tf.compat.v1.placeholder(tf.float32)
    # with tf.compat.v1.Session() as sess:
    #     sess.run(rmspe.ref(), feed_dict={thing: y_pred})
    # return thing

def ann():
    model = Sequential()
    model.add(Dense(64, input_dim=14,kernel_initializer=RandomNormal, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='tanh'))
    model.compile(loss=root_mean_squared_percent_error, optimizer='adam', metrics=['mse',tf.keras.metrics.MeanAbsolutePercentageError(),tf.keras.metrics.MeanAbsoluteError()])
    return model

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.33)


# rmspeScore = make_scorer(root_mean_squared_percent_error, greater_is_better=False)
model = KerasRegressor(build_fn=ann, epochs=100, batch_size=32, verbose=1 )
kfold = KFold(n_splits = 5)
Y_train = Y_train.values.reshape(Y_train.size,1)
results = cross_val_score(model, X_train,Y_train, cv=kfold) #scoring=rmspeScore)
print("Wider: %.2f RMSPE (%.2f) Std deviation " % (results.mean(), results.std()))

# model.fit(X_train,Y_train)
# a = model.predict(X_test)
# print("RMSE for Deep Network:",np.sqrt(np.mean((Y_test-a.reshape(a.size,))**2)))
