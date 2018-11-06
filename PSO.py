#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:30:51 2018

@author: sanjeethrao
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import model_selection
from sklearn import preprocessing


import pyswarms as ps
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt


from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters.formatters import Designer
from matplotlib import animation
from matplotlib import rc
from IPython.display import HTML
# Import PySwarms


from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

# Import PySwarms
# Import modules



from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

# Load the dataset any data set

region = "A"
filename = region + ".csv"
#feature columns for x coordinates
x= np.array(pd.read_csv(filename, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]))
#feature colums for y coordinates
y = np.array(pd.read_csv(filename, usecols=[40]))





#normalizing the data
normalized_X = preprocessing.scale(x)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
normalized_X, y, test_size=0.10, shuffle=False, stratify=None)


Y_train_new = Y_train.ravel()


le = preprocessing.LabelEncoder()
le.fit(Y_train_new)
le.classes_
Y_train_transformed = le.transform(Y_train_new)
le.inverse_transform(le.transform(Y_train_new))


# Forward propagation
def forward_prop(params):
    # Neural network architecture
    n_inputs = X_train.shape[1]
    n_hidden = 1
    n_classes = 2
    p=(n_inputs*n_hidden)
    q=((n_inputs*n_hidden)+n_hidden)
    r=(((n_inputs*n_hidden)+n_hidden)+(n_hidden*n_classes))
    s=((((n_inputs*n_hidden)+n_hidden)+(n_hidden*n_classes))+n_classes)
    # Roll-back the weights and biases
    W1 = params[0:p].reshape((n_inputs,n_hidden))
    b1 = params[p:q].reshape((n_hidden,))
    W2 = params[q:r].reshape((n_hidden,n_classes))
    b2 = params[r:s].reshape((n_classes,))
    
    # Perform forward propagation
    z1 = X_train.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood
    N = Y_train.shape[0] # Number of samples
    correct_logprobs = -np.log(probs[range(N), Y_train])
    loss = np.sum(correct_logprobs) / N
    
    return loss

    
def f(X_train, X_test,alpha=0.88):
    n_particles = 100
    j = [forward_prop(X_train[i]) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm
options = {'c1': 0.1, 'c2': 0.01, 'w':0.9}

# Call instance of PSO
dimensions = (X_train.shape[1] * 1) + (1 * 2) + 1 + 2#(  (X_train.shape[1] * n_hidden) + (n_hidden* n_classes) + n_hidden+ n_classes   )
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)
print(dimensions)
print(optimizer)
# Perform optimization
cost, pos = optimizer.optimize(f, print_step=100, iters=10, verbose=3)

def predict(X_train, pos):
    # Neural network architecture
    n_inputs = X_train.shape[1]
    n_hidden = 0
    n_classes = 2
    p=(n_inputs*n_hidden)
    q=((n_inputs*n_hidden)+n_hidden)
    r=(((n_inputs*n_hidden)+n_hidden)+(n_hidden*n_classes))
    s=((((n_inputs*n_hidden)+n_hidden)+(n_hidden*n_classes))+n_classes)
    
    # Roll-back the weights and biases
    W1 = pos[0:p].reshape((n_inputs,n_hidden))
    b1 = pos[p:q].reshape((n_hidden,))
    W2 = pos[q:r].reshape((n_hidden,n_classes))
    b2 = pos[r:s].reshape((n_classes,))

    # Perform forward propagation
    z1 = X_train.dot(W1) + b1   # Pre-activation in Layer 1
    a1 = np.tanh(z1)            # Activation in Layer 1
    z2 = a1.dot(W2) + b2        # Pre-activation in Layer 2
    logits = z2                 # Logits for Layer 2

    y_pred = np.argmax(logits, axis=1)
    
    return y_pred

A=(predict(X_train, pos) == Y_train).mean()
print('=================================================')
print('=================================================')
print('=================================================')
print("_____________________Accurecy____________________")
print('|\t\t',100*A,'%\t\t|')
print("_________________________________________________")
print('=================================================')
print('=================================================')





