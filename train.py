# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:21:15 2020

@author: RikSo
"""


import keras
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


#asignar la data de entrenamiento
data = pd.read_csv(r"D:\data\fit.csv",sep=';')
#train = pd.read_csv('D:\SublimeText\Data\Local\PYTHON\Hackaton\dataset\target.csv')
predictores = data.iloc[:,[2,3,4,5,6,7,8,9]]

n_cols = predictores.shape[1]
print(n_cols)

# Convert the target to categorical: target
target = to_categorical(data.Survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))

# Add the output layer
#softmax calcula una probabilidades para cada clase.
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(predictores,target)
