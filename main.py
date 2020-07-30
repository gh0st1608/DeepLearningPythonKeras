# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:50:49 2020

@author: RikSo
"""


import numpy as np
import matplotlib.pyplot as plt
import funciones as func
from sklearn.metrics import mean_squared_error


input_data = np.array([1,2,3])
num_actualizaciones = 6
mse_hist = []
pred_vect = []
target = 0
target_vect = np.array([3,3,3,3,3])
weights = np.array([0,2,1])

print("This network does not have any hidden layers, and it goes directly ")
print("from the input (with 3 nodes) to an output node. Note that weights ")
print("is a single array.")
      
for i in range(1,num_actualizaciones):
    # Calculate the slope: slope
    slope = func.get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights  - 0.01 * slope
    
    # Calcular la prediccion 
    predict = func.prediccion(input_data,weights)
    
    # Agregar la prediccion a un vector
    pred_vect.append(predict)
    
    # Calculate mse with new weights: mse
    mse =func.get_mse(predict, target)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)
    
    print("Iteracion: " + str(i))
    print("Valor Predecido: " + str(predict))
    print("Valor Pendiente: " + str(slope))
    print("Valor mse: " + str(mse))
    print("\n")
    

# Mostar la metrica    
mse_0 = mean_squared_error(target_vect,pred_vect)
print("Mean squared error with weights_0: %f" %mse_0)
# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()







