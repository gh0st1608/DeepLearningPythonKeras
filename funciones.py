# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:53:49 2020

@author: RikSo
"""


def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    
    # Return the value just calculated
    return(output)


def get_slope(input_data,target, weights):
    
    preds = (weights * input_data).sum()

    # Calculate the error: error
    error = preds - target

    # Calculate the slope: slope
    slope = 2 * input_data * error
    
    return slope


def prediccion(input_data,weights_updated):
    preds_updated = (weights_updated * input_data).sum()
    
    return preds_updated


def get_mse(preds_updated,target):
    mse = preds_updated - target
    
    return mse