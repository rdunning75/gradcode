import pandas as pd
from numpy import mean
from numpy import std
import keras as ks
import sklearn as skl
import tensorflow as tf
from sklearn.linear_model import  LinearRegression as LR
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import  LogisticRegression as LogR
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from keras.models import Sequential
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
# print(X)
# print(df["FIRE_SIZE_CLASS"].unique())
df["FIRE_SIZE_CLASS"] = le.fit_transform(df["FIRE_SIZE_CLASS"])
# y = LabelBinarizer().fit_transform(df["FIRE_SIZE_CLASS"])

y = df["FIRE_SIZE_CLASS"]

# print(y)


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#test to get this working

model = Sequential()
model.add(ks.Input(shape=(14,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16,  activation='tanh'))

model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=50, batch_size=10)
print("working")
_,accuracy = model.evaluate(x_test,y_test)
print("working")

print('Accuracy: %.2f' %(accuracy*100))


# y_train = LabelBinarizer().fit_transform(y_train)
# y_test = LabelBinarizer().fit_transform(y_test)
# linearRegression = LR()
#
#
# linearRegression.fit(x_train,y_train)
# predictions = linearRegression.predict(x_test)
# score_train = linearRegression.score(x_train,y_train)
# score_test = linearRegression.score(x_test,y_test)
#
# print(score_test)
# print(score_train)
#
# #  solvers: lbfgs, saga, sag
# logisticregresion = LogR(multi_class='multinomial', solver='newton-cg')
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(logisticregresion, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
# print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores)))
#



