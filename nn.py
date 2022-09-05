
import numpy as np
import pandas as pd
from nn_helper_func import *
from sklearn






class NN():

    cache = dict({})
    layers = []
    total_parms = 0

    def add_layer(self,L):
        self.layers.append(L)
        self.model_compiled = False



    def compile(self):
        prev_N = self.input_n
        for _i, layer in enumerate(self.layers):
            #each layer has the dimension of (hidden_neurons x hidden_neurons_prev.Layer)
            layer.parameters["W"] = np.zeros((layer.n_neurons, prev_N))
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
        x_new = x
        #pass all the inputs directly through all the Layers
        for _i,layer in enumerate(self.layers):
            #pass it through the layers....
            W = layer.params["W"]
            b = layer.params["b"]
            out = W@x_new + b


            #now the activation
            if layer.activation == "Relu":
                a = relu(out)
            elif layer.activation == "sigmoid":
                a = sigmoid(out)
            elif layer.activation == "tanh":
                a = tanh(out)


            layer.params["a"] = a

            x_new = a


        print("done")





    def compute_cost(self):
        m = self.Y.shape[1]


        a_last = self.layers[-1].params["a"]
        # Compute loss from aL and y.
        # (≈ 1 lines of code)
        # cost = ...
        # YOUR CODE STARTS HERE
        cost = -1 / m * np.sum(self.Y * np.log(a_last) + (1 - self.Y) * np.log(1 - a_last), axis=1)

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost









    def __init__(self,input_size = (1,1)):
        self.input_size = input_size
        self.input_n = np.product(self.input_size)








class DenseLayer():
    def __init__(self,n_neurons = 10,activation = "ReLu"):
        super().__init__()
        self.n_neurons = n_neurons
        self.parameters = dict({})
        self.activation = activation

    def step(self,x):

        #get the parameters
        pass




















if __name__ == '__main__':


    data_train = pd.read_csv("train.csv")
    data_test = pd.read_csv("test.csv")

    data_in = pd.DataFrame([1])

    myNetwork = NN(input_size = (28,28))
    Layer1 = DenseLayer(n_neurons=100)
    Layer2 = DenseLayer(n_neurons=10)

    myNetwork.add_layer(Layer1)
    myNetwork.add_layer(Layer2)
    myNetwork.compile()






    print("done")








