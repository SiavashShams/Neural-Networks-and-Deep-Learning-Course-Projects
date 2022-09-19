# -*- coding: utf-8 -*-
"""HW2_Part2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MpUmyiKndM3olY00OPK_2JNewv__NsMK
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import keras
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
drive.mount('/content/drive')
# %cd /content/drive/My Drive/HW2_deep/

"""# **PART 1**



"""

df = pd.read_csv("data.csv")

# plot distribution of data
fig = plt.figure(figsize=(10,7))
fig.add_subplot(2,1,1)
sns.distplot(df['price'])
fig.add_subplot(2,1,2)
sns.boxplot(df['price'])
plt.tight_layout()

# remove outliers
df=df[df["price"]<=10000000]

df.head()

# date column to year/month/day
print(df.shape)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)
df['day']=df['date'].apply(lambda date:date.day)
y = df.price
X = df.drop(columns=["price"], axis=1)

df=df.drop("date",axis=1)

cat_cols=["street", "city","statezip","country"]
X_categorical_df = pd.get_dummies(X[cat_cols], columns=cat_cols)

y = df.price
X = df.drop(columns=["price"], axis=1)
num_cols=list(df.columns.values)
num_cols=list(set(num_cols)-set(cat_cols))
num_cols.remove("price")
print(num_cols)
X_final = X[num_cols]
X_final = X_final.join(X_categorical_df)

from sklearn import preprocessing
X_final[num_cols] = preprocessing.StandardScaler().fit_transform(X_final[num_cols])
#X_final[num_cols] = MinMaxScaler().fit_transform(X_final[num_cols])

X_final.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=420)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape,y_val.shape, X_val.shape

import tensorflow as tf
from keras import layers, models
from tensorflow.keras.optimizers import Adam

# define our model
model = models.Sequential()
model.add(layers.Dense(60,activation='relu'))
model.add(layers.Dense(60,activation='relu'))
model.add(layers.Dense(60,activation='relu'))
#model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='Adam',loss='mse')
model.fit(x=X_train,y=y_train,
          validation_data=(X_val,y_val),
          batch_size=128,epochs=200)
model.summary()
loss_df = pd.DataFrame(model.history.history)

loss_df.plot(figsize=(12,8))
plt.xlabel("Epoch")
plt.ylabel("Loss")
y_pred = model.predict(X_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))# Visualizing Our predictions

"""#PART 2"""

model = models.Sequential()
model.add(layers.Dense(60,activation='relu'))
model.add(layers.Dense(60,activation='relu'))
model.add(layers.Dense(60,activation='relu'))
#model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='Adam',loss='mse',metrics=["mse","mae"])
history = model.fit(x=X_train,y=y_train,
          validation_data=(X_val,y_val),
          batch_size=128,epochs=200)
model.summary()
loss_df = pd.DataFrame(model.history.history)

y_pred = model.predict(X_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))# Visualizing Our predictions
plt.plot(history.history['mse'], label='mse')
plt.plot(history.history['val_mse'], label='val_mse')
plt.title('MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
# Perfect predictions
plt.plot(y_test,y_test,'r')
plt.title('Predicted and Real Prices')
plt.xlabel('num_house')
plt.ylabel('Price')
plt.legend(["Real","Predicted"])

"""# PART 3"""

model = models.Sequential()
model.add(layers.Dense(60,activation='relu'))
model.add(layers.Dense(60,activation='relu'))
model.add(layers.Dense(60,activation='relu'))
#model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='Adam',loss='mae',metrics=["mse","mae"])
history = model.fit(x=X_train,y=y_train,
          validation_data=(X_val,y_val),
          batch_size=128,epochs=200)
model.summary()
loss_df = pd.DataFrame(model.history.history)

y_pred = model.predict(X_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))# Visualizing Our predictions
plt.plot(history.history['mse'], label='mse')
plt.plot(history.history['val_mse'], label='val_mse')
plt.title('MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
# Perfect predictions
plt.plot(y_test,y_test,'r')
plt.title('Predicted and Real Prices')
plt.xlabel('num_house')
plt.ylabel('Price')
plt.legend(["Real","Predicted"])

"""# BONUS"""

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)

y_pred_ridge=clf.predict(X_test)
y_pred_reg=reg.predict(X_test)

sc=clf.score(X_test, y_test, sample_weight=None)

from sklearn import metrics
print('Ridge MAE:', metrics.mean_absolute_error(y_test, y_pred_ridge))  
print('Ridge MSE:', metrics.mean_squared_error(y_test, y_pred_ridge))  
print('Ridge RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge)))
print('Ridge VarScore:',metrics.explained_variance_score(y_test,y_pred_ridge))# Visualizing Our predictions
print('\nLinear Regression MAE:', metrics.mean_absolute_error(y_test, y_pred_reg))  
print('Linear Regression MSE:', metrics.mean_squared_error(y_test, y_pred_reg))  
print('Linear Regression RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)))
print('Linear Regression VarScore:',metrics.explained_variance_score(y_test,y_pred_reg))# Visualizing Our predictions