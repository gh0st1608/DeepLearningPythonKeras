# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:49:44 2020

@author: RikSo
"""

"""
Specifying a model
Now you'll get to work with your first model in Keras, 
and will immediately be able to run more complex neural network 
models on larger datasets compared to the first two chapters.

To start, you'll take the skeleton of a neural network 
and add a hidden layer and an output layer. You'll then fit that model
and see Keras do the optimization so your model continually gets better.

As a start, you'll predict workers wages based on characteristics 
like their industry, education and level of experience. 
You can find the dataset in a pandas dataframe called df. 
For convenience, everything in df except for the target has been converted 
to a NumPy matrix called predictors. The target, wage_per_hour, 
is available as a NumPy matrix called target.

For all exercises in this chapter, we've imported the Sequential model constructor, 
the Dense layer constructor, and pandas.
"""

#dataframe de trabajo = df
#predictors = df.to_numpy() -> transforma un dataframe a un arreglo NumPy
#list(df) -> ver cabeceras de un dataframe
# Import necessary modules
import keras
import pandas
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50,activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32,activation='relu'))

# Add the output layer
model.add(Dense(1))


"""
Compiling the model
You're now going to compile the model you specified earlier. 
To compile the model, you need to specify the optimizer and 
loss function to use. In the video, Dan mentioned that the Adam optimizer 
is an excellent choice. You can read more about it as well as other 
keras optimizers here, and if you are really curious to learn more, 
you can read the original paper that introduced the Adam optimizer.

In this exercise, you'll use the Adam optimizer and the mean squared error 
loss function. Go for it!
"""
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer = 'adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

"""
Fitting the model
You're at the most fun part. You'll now fit the model. 
Recall that the data to be used as predictive features is loaded in a 
NumPy matrix called predictors and the data to be predicted is stored in a 
NumPy matrix called target. Your model is pre-written and it has been compiled 
with the code from the previous exercise.
"""
#dataframe = df
#target

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors,target)



"""
Last steps in classification models

You'll now create a classification model using the titanic dataset, 
which has been pre-loaded into a DataFrame called df. You'll take information 
about the passengers and predict which ones survived.

The predictive variables are stored in a NumPy array predictors. 
The target to predict is in df.survived, though you'll have to manipulate 
it for keras. The number of predictive features is stored in n_cols.

Here, you'll use the 'sgd' optimizer, which stands for Stochastic Gradient Descent. 
You'll learn more about this in the next chapter!
"""



# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(predictors,target)


"""
Making predictions
The trained network from your previous coding exercise is now stored as model. 
New data to make predictions is stored in a NumPy array as pred_data. 
Use model to make predictions on your new data.

In this exercise, your predictions will be probabilities, 
which is the most common way for data scientists to communicate 
their predictions to colleagues.
"""

# Specify, compile, and fit the model
'print(pred_data)
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(predictors, target)


#pred_data es un nuevo dataset para predecir
# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)










