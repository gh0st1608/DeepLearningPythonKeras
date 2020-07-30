# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:35:22 2020

@author: RikSo
"""

#librerias
import numpy as np
import matplotlib.pyplot as plt

"""
Each data point is a customer. The first input is how many accounts 
they have, and the second input is how many children they have. 
The model will predict how many transactions the user makes in the 
next year. You will use this data throughout the first 2 chapters 
of this course.

The input data has been pre-loaded as input_data, and the weights 
are available in a dictionary called weights. The array of weights 
for the first node in the hidden layer are in weights['node_0'], 
and the array of weights for the second node in the hidden layer 
are in weights['node_1'].

The weights feeding into the output node are available in weights['output'].

NumPy will be pre-imported for you as np in all exercises.
"""

#inputs
input_data=np.array([2,3])

#pesos
weights={'node_0':np.array([1,1]),'node_1':np.array([-1,1]),'output':np.array([2,1])}

#calculo de los valores de los nodos
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

#colocar el valor de los nodos en la capa oculta
hidden_layer_values = np.array([node_0_value,node_1_value])
print("El primer valor de las capa oculta es: " + str(hidden_layer_values[0]))
print("El segundo valor de las capa oculta es: " + str(hidden_layer_values[1]))

#calcular las salidas
output = (hidden_layer_values * weights['output']).sum()
print("La salida es :" + str(output))


"""
The Rectified Linear Activation Function
As Dan explained to you in the video, an "activation function" 
is a function applied at each node. It converts the node's input 
into some output.

The rectified linear activation function (called ReLU) has been 
shown to lead to very high-performance networks. This function 
takes a single number as an input, returning 0 if the input is 
negative, and the input if the input is positive.

Here are some examples:
relu(3) = 3
relu(-3) = 0
"""


def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print("El valor de la salida del modelo aplicando Funcion de Activacion es: " + str(model_output))

"""
Applying the network to many observations/rows of data

You'll now define a function called predict_with_network() 
which will generate predictions for multiple data observations, 
which are pre-loaded as input_data. As before, 
weights are also pre-loaded. In addition, the relu() function 
you defined in the previous exercise has been pre-loaded.
"""

# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    model_output = (hidden_layer_outputs * weights['output']).sum()
    
    # Return model output
    return(model_output)

# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)


"""
Multi-layer neural networks
In this exercise, you'll write code to do forward propagation 
for a neural network with 2 hidden layers. Each hidden layer 
has two nodes. The input data has been preloaded as input_data. 
The nodes in the first hidden layer are called node_0_0 and node_0_1. 
Their weights are pre-loaded as weights['node_0_0'] and weights['node_0_1'] 
respectively.

The nodes in the second hidden layer are called node_1_0 and node_1_1. 
Their weights are pre-loaded as weights['node_1_0'] and weights['node_1_1'] 
respectively.

We then create a model output from the hidden nodes using weights pre-loaded 
as weights['output'].
"""

#input_data = [3,5]
#weights =  {'node_0_1':array([4,-5]),'node_1_1':array([1,2]),
#           'node_0_0':array([2,4]),'node_1_0':array([-1,2]),'output':array([2,7])}
def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)
#la ultima capa caputa las caracteristicas mas complejas
#el modelo asigna los pesos que optimizan el objetivo


"""
Coding how weight changes affect accuracy
Now you'll get to change weights in a real network and see 
how they affect model accuracy
Its weights have been pre-loaded as weights_0. Your task in 
this exercise is to update a single weight in weights_0 
to create weights_1, which gives a perfect prediction 
(in which the predicted value is equal to target_actual: 3).

Use a pen and paper if necessary to experiment with different combinations. 
You'll use the predict_with_network() function, which takes an array of data 
as the first argument, and weights as the second argument.
"""

# The data point you will make a prediction for
input_data = np.array([0, 3]) 

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [2, 2]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)

"""
Scaling up to multiple data points

You've seen how different weights will have different accuracies 
on a single prediction. But usually, you'll want to measure model 
accuracy on many points. You'll now write code to compare model 
accuracies for two different sets of weights, which have been 
stored as weights_0 and weights_1.

input_data is a list of arrays. Each item in that list contains 
the data to make a single prediction. target_actuals is a list of numbers. 
Each item in that list is the actual value we are trying to predict.

In this exercise, you'll use the mean_squared_error() function 
from sklearn.metrics. It takes the true values and the predicted 
values as arguments.

You'll also use the preloaded predict_with_network() function, 
which takes an array of data as the first argument, and weights 
as the second argument.
"""

from sklearn.metrics import mean_squared_error
target_actuals = 3
# Create model_output_0 
model_output_0 = []
# Create model_output_1
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row,weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row,weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals,model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals,model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)

"""
Calculating slopes

You're now going to practice calculating slopes.
 When plotting the mean-squared error loss function against predictions, 
 the slope is 2 * x * (xb-y), or 2 * input_data * error. Note that x 
 and b may have multiple numbers (x is a vector for each data point, 
 and b is a vector). In this case, the output will also be a vector, 
 which is exactly what you want.

You're ready to write the code to calculate this slope while 
using a single data point. You'll use pre-defined weights 
ºcalled weights as well as data for a single point called input_data. 
The actual value of the target you want to predict is stored in target.

"""
target = 0
# Calculate the predictions: preds
preds = (input_data * weights).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = input_data * error  * 2

# Print the slope
print(slope)

"""
Improving model weights

Hurray! You've just calculated the slopes you need. 
Now it's time to use those slopes to improve your model. 
If you add the slopes to your weights, you will move in the right direction. 
However, it's possible to move too far in that direction. 
So you will want to take a small step in that direction first, 
using a lower learning rate, and verify that the model is improving.

The weights have been pre-loaded as weights, the actual value of the target as target, and the input data as input_data. The predictions from the initial weights are stored as preds.

"""
#input_data = [1,2,3]
#weights = [0,2,1]
#target = 0


# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: 
weights_updated = weights - learning_rate * slope

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)

"""
Making multiple updates to weights

You're now going to make multiple updates so you can dramatically 
improve your model weights, and see how the predictions improve with each update.
To keep your code clean, there is a pre-loaded get_slope() function 
that takes input_data, target, and weights as arguments. 
There is also a get_mse() function that takes the same arguments. 
The input_data, target, and weights have been pre-loaded.

This network does not have any hidden layers, and it goes directly 
from the input (with 3 nodes) to an output node. Note that weights 
is a single array.

We have also pre-loaded matplotlib.pyplot, and the error history will be 
plotted after you have done your gradient descent steps.
"""
#funcion predefinida
def get_slope(input_data,target, weights):
    
    preds = (weights * input_data).sum()

    # Calculate the error: error
    error = preds - target

    # Calculate the slope: slope
    slope = 2 * input_data * error
    
    return slope


def get_mse(input_data,target,weights):
    
    preds_updated = (weights_updated * input_data).sum()
    mse = preds_updated - target
    
    return mse



n_updates = 20
mse_hist = []
target = 0
#weights = [0,2,1]

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights  - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()
























