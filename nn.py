
import numpy as np
import pandas as pd
from nn_helper_func import *

from sklearn.preprocessing import OneHotEncoder







class NN():

    cache = dict({})
    layers = []
    total_parms = 0
    def __init__(self,input_size = (1,1),
                 learning_rate = 0.01):
        self.input_size = input_size
        self.input_n = np.product(self.input_size)

    def add_layer(self,L):
        self.layers.append(L)
        self.model_compiled = False

    def compile(self):
        prev_N = self.input_n
        for _i, layer in enumerate(self.layers):
            #assign name
            layer.name = layer.activation + "_" + str(_i+1)

            #each layer has the dimension of (hidden_neurons x hidden_neurons_prev.Layer)
            layer.parameters["W"] = np.random.randn(layer.n_neurons, prev_N) *0.01
            layer.parameters["b"] = np.zeros((layer.n_neurons, 1))

            layer.parameters["a"] = np.zeros((layer.n_neurons, 1))

            layer.n_params = int(np.product(layer.parameters["W"].shape) + np.product(layer.parameters["b"]))

            self.total_parms += layer.n_params

            print(30*"*")
            print("Layer: {:d}".format(_i+1) + ": ", layer.activation)
            print("W's shape:", layer.parameters["W"].shape)
            print("b's shape:", layer.parameters["b"].shape)
            print("")
            prev_N = layer.n_neurons

        print("Total Params: {:d}".format(self.total_parms))
        self.model_compiled = True




    def forward_pass(self,x):

        if not self.model_compiled:
            raise BaseException()



        x_new = x
        #pass all the inputs directly through all the Layers
        for _i,layer in enumerate(self.layers):
            #pass it through the layers....
            W = layer.parameters["W"]
            b = layer.parameters["b"]
            Z = W@x_new + b

            #Store the parameters of the linear function in Var. Z
            layer.parameters["Z"] = Z

            #now the activation
            if layer.activation == "ReLu":
                a = relu(Z)
            elif layer.activation == "sigmoid":
                a = sigmoid(Z)
            elif layer.activation == "tanh":
                a = tanh(Z)


            layer.parameters["a"] = a

            x_new = a


        print("done")





    def compute_cost(self, y):
        m = y.shape[1]


        a_last = self.layers[-1].parameters["a"]
        # Compute loss from aL and y.
        # (≈ 1 lines of code)
        # cost = ...
        # YOUR CODE STARTS HERE
        cost = -1 / m * np.sum(y * np.log(a_last) + (1 - y) * np.log(1 - a_last))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        print("Cost:",cost)
        return cost


    def backward_step(self,Y):
        """
        Implementation of a backward calculation step

        :return:
        """

        #Derviative of cost function with respect to A
        dAL = - (np.divide(self.Y, self.layers[-1].parameters["a"]) -
                 np.divide(1 - self.Y, 1 - self.layers[-1].parameters["a"]))

        activation_backward(dAL,self.layers[-1])
        prevLayer = self.layers[-1]
        for _i,layer in enumerate(reversed(self.layers[:-1])):
            print(_i,layer.activation)
            dA_prev= prevLayer.parameters["dA"]
            activation_backward(dA_prev, self.layers[-1])

            prevLayer = layer










class DenseLayer():
    def __init__(self,n_neurons = 10,activation = "ReLu"):
        self.n_neurons = n_neurons
        self.parameters = dict({})
        self.activation = activation

    def step(self,x)

        #get the parameters
        pass























if __name__ == '__main__':


    data_train = pd.read_csv(r"data\fashion-mnist_train.csv",nrows=1000)
    data_test = pd.read_csv(r"data\fashion-mnist_test.csv")

    ylabs_train = data_train.pop("label")
    ylabs_test = data_test.pop("label")


    encoder = OneHotEncoder(sparse=False)

    Y_train = encoder.fit_transform(ylabs_train.values.reshape(-1,1)).T
    Y_test = encoder.transform(ylabs_test.values.reshape(-1,1)).T





    myNetwork = NN(input_size = (28,28))
    Layer1 = DenseLayer(n_neurons=100)
    Layer2 = DenseLayer(n_neurons=10,activation="sigmoid")

    myNetwork.add_layer(Layer1)
    myNetwork.add_layer(Layer2)
    myNetwork.compile()



    #do forward pass
    myNetwork.forward_pass(data_train.values.T)

    myNetwork.compute_cost(Y_train)
    myNetwork.Y = Y_train


    myNetwork.backward_step()





    print("done")








