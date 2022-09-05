

import numpy as np


def relu(Z):
    A = np.maximum(Z,0)
    return A

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return ( A )

def tanh(Z):
    A = np.tanh(Z)
    return A

def forward_pass(Model):
    pass


