#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[7]:


df = pd.read_csv("C:\\Users\\Pooja R\Desktop\projects\\Tesla_stocks.csv", parse_dates = ['Date'])


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


train_length = round(len(df)*0.7)
lg = len(df)
val_length = lg-train_length

print('Total observations:',lg)
print('Training set:', train_length)
print('Validation set:', val_length)


# In[11]:


train_data = df['Open'][:train_length,]
val_data = df['Open'][train_length:,]


# In[12]:


val_data


# In[13]:


train_data


# In[14]:


train=train_data.values.reshape(-1,1)
train


# In[15]:


# Normalization
# Scale the data between the range [0,1]

scaler = MinMaxScaler()
scaled_trainset = scaler.fit_transform(train)


# In[16]:


scaled_trainset


# In[17]:


plt.subplots(figsize = (15,6))
plt.plot(train)
plt.show()


# In[18]:


x_train = []
y_train = []
step = 50

for i in range(step, train_length):
    x_train.append(scaled_trainset[i-step:i,0])
    y_train.append(scaled_trainset[i,0])


# In[19]:


X_train, y_train = np.array(x_train), np.array(y_train)


# In[20]:


X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
y_train.reshape(y_train.shape[0],1)


# In[21]:


print(X_train.shape)
print(y_train.shape)


# In[22]:


X_train[0].shape


# In[23]:


y_train[0]


# In[24]:


# RNN model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout


# In[25]:


model = Sequential()

model.add(
    SimpleRNN(units = 50,return_sequences= True,input_shape = (X_train.shape[1],1)))

model.add(
    Dropout(0.2))

model.add(
    SimpleRNN(units = 50, return_sequences = True)
             )

model.add(
    Dropout(0.2)
             )

model.add(
    SimpleRNN(units = 50, return_sequences = True)
             )

model.add(
    Dropout(0.2)
             )

model.add(
    SimpleRNN(units = 50)
             )

model.add(
    Dropout(0.2)
             )

model.add(
    Dense(units = 1))


# In[26]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])


# In[27]:


model.summary()


# In[28]:


history = model.fit(X_train, y_train, epochs = 50, batch_size =32)


# In[29]:


plt.plot(history.history['loss'])


# In[30]:


plt.plot(history.history['accuracy'])


# In[31]:


y_pred = model.predict(X_train)
y_pred = scaler.inverse_transform(y_pred.reshape(1,-1))


# In[32]:


y_pred


# In[33]:


y_train


# In[34]:


y_train = scaler.inverse_transform(y_train.reshape(1,-1))
y_train


# In[35]:


y_train.shape
y_train = np.reshape(y_train, (1134,1))


# In[36]:


y_pred.shape
y_pred = np.reshape(y_pred,(1134,1))


# In[37]:


plt.figure(figsize = (30,10))
plt.plot(y_pred, ls = '--', label = 'y_pred', lw = 2)
plt.plot(y_train, label = 'y_train')
plt.legend()
plt.show()


# In[38]:


val = val_data.values.reshape(-1,1)
val


# In[39]:


scaled_valset = scaler.fit_transform(val)


# In[40]:


xval_train = []
yval_train = []
step = 50

for i in range(step, val_length):
    xval_train.append(scaled_valset[i-step:i,0])
    yval_train.append(scaled_valset[i,0])


# In[41]:


X_val, y_val = np.array(xval_train), np.array(yval_train)


# In[42]:


X_val = np.reshape(X_val, (X_val.shape[0],X_val.shape[1],1))  # reshape to 3D array
y_val = np.reshape(y_val, (-1,1))


# In[43]:


y_pred_val = model.predict(X_val)


# In[44]:


y_pred_val = scaler.inverse_transform(y_pred_val)


# In[45]:


y_val_is = scaler.inverse_transform(y_val)


# In[46]:


plt.figure(figsize = (30,10))
plt.plot(y_pred_val, label = 'y_pred')
plt.plot(y_val_is, label = 'y_val')
plt.legend()
plt.show()


# In[47]:


import tensorflow as tf
import os
from tensorflow.keras.models import load_model


# In[48]:


model.save(os.path.join('model', 'SimpleRNN_Forecasting.h5'))
new_model = load_model(os.path.join('model', 'SimpleRNN_Forecasting.h5'))


# In[ ]:




