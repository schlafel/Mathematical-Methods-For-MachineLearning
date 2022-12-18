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

def relu_backward(dA, l):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """


    Z = l.parameters["Z"]
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, l):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = l.parameters["Z"]
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)


    return dZ


def tanh_backward(dA,l):
    Z = l.parameters["Z"]
    dZ = 1 - dA**2
    assert (dZ.shape == Z.shape)

    return dZ

def linear_backward(dZ,l):
    A, W, b = l.parameters["a_prev"],l.parameters["W"],l.parameters["b"]

    m = A.shape[1]


    dW = 1/m * dZ@A.T
    db = np.sum(dZ,axis = 1,keepdims = True)
    dA_prev = W.T@dZ


    l.parameters["dW"] = dW
    l.parameters["db"] = db
    l.parameters["dA"] = dA_prev




def activation_backward(dA,l):


    #first calculate the derivative of dA with respect to Z to get dZ
    if l.activation == "ReLu":
        dZ = relu_backward(dA,l)
    elif l.activation == "sigmoid":
        dZ = sigmoid_backward(dA,l)
    elif l.activation == "tanh":
        dZ = sigmoid_backward(dA,l)
    else:
        raise ValueError("Activation is not known... Got {}, expected one of the following: ".format(l.activation),
                         ["ReLu","sigmoid","tanh"])



    #then we have the linear_backward activation gradients to calculate
    linear_backward(dZ,l)





