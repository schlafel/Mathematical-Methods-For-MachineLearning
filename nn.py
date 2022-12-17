
import numpy as np
import pandas as pd
from nn_helper_func import *
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt





class NN():

    cache = dict({})
    layers = []
    total_parms = 0
    def __init__(self,input_size = (1,1),
                 learning_rate = 0.01):
        self.input_size = input_size
        self.input_n = np.product(self.input_size)
        self.learning_rate = learning_rate
        self.layers = []
        self.parameters = dict()

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


    def predict(self,x):
        preds = self.forward_pass(x,training=False)

        return preds
    def forward_pass(self,x,training = True):

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
            if training:
                layer.parameters["Z"] = Z

            #now the activation
            if layer.activation == "ReLu":
                a = relu(Z)
            elif layer.activation == "sigmoid":
                a = sigmoid(Z)
            elif layer.activation == "tanh":
                a = tanh(Z)

            if training:
                layer.parameters["a"] = a
                layer.parameters["a_prev"] = x_new

            x_new = a


        print("done")
        return x_new





    def compute_cost(self, y):
        m = y.shape[1]


        a_last = self.layers[-1].parameters["a"]
        # Compute loss from aL and y.


        cost = -1 / m * np.sum(y * np.log(a_last) + (1 - y) * np.log(1 - a_last))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        print("Cost:",cost)
        return cost


    def backward_step(self):
        """
        Implementation of a backward calculation step

        :return:
        """

        #Derviative of cost function with respect to A
        dAL = - (np.divide(self.Y, self.layers[-1].parameters["a"]) -
                 np.divide(1 - self.Y, 1 - self.layers[-1].parameters["a"]))

        activation_backward(dAL,self.layers[-1])
        self.update_parameters(self.layers[-1])


        prevLayer = self.layers[-1]
        for _i,layer in enumerate(reversed(self.layers[:-1])):
            print(_i,layer.activation)
            dA_prev= prevLayer.parameters["dA"]
            activation_backward(dA_prev, layer)

            self.update_parameters(layer)

            prevLayer = layer





    def update_parameters(self, layer):
        layer.parameters["W"] = layer.parameters["W"] -  layer.parameters["dW"]*self.learning_rate
        layer.parameters["b"] = layer.parameters["b"] -  layer.parameters["db"]*self.learning_rate



    def save(self,path):

        with open(path, 'wb') as f:
            pickle.dump(self, f)


class DenseLayer():
    def __init__(self,n_neurons = 10,activation = "ReLu"):
        self.n_neurons = n_neurons
        self.parameters = dict({})
        self.activation = activation

    def step(self,x):

        #get the parameters
        pass


def load_data():
    train_data = np.loadtxt(r"data/fashion-mnist_train.csv", delimiter=",",skiprows=1)
    Y_train = train_data[:,0]
    X_train = train_data[:,1:]

    test_data = np.loadtxt(r"data/fashion-mnist_test.csv", delimiter=",",skiprows=1)
    Y_test = test_data[:,0]
    X_test = test_data[:,1:]

    # data_train = pd.read_csv(r"data\fashion-mnist_train.csv", nrows=100000000)
    # data_test = pd.read_csv(r"data\fashion-mnist_test.csv")
    # ylabs_train = data_train.pop("label")
    # ylabs_test = data_test.pop("label")
    encoder = OneHotEncoder(sparse=False)
    Y_train_enc = encoder.fit_transform(Y_train.reshape(-1, 1)).T
    Y_test_enc = encoder.transform(Y_test.reshape(-1, 1)).T

    return X_train,Y_train_enc,X_test,Y_test_enc,encoder

if __name__ == '__main__':


    data_train,Y_train,data_test,Y_test,enc = load_data()

    myNetwork = NN(input_size = (28,28),
                   learning_rate = 1e-4)
    Layer1 = DenseLayer(n_neurons=100)
    Layer2 = DenseLayer(n_neurons=10,activation="sigmoid")

    myNetwork.add_layer(Layer1)
    myNetwork.add_layer(Layer2)
    myNetwork.compile()



    #do forward pass
    x_new = myNetwork.forward_pass(data_train.T)

    myNetwork.compute_cost(Y_train)
    myNetwork.Y = Y_train

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, =ax.plot(0,0)

    train_ = []

    for i in range(1,30):
        print("Epoch:", i, 30*"*")
        myNetwork.backward_step()
        myNetwork.forward_pass(data_train.T)
        loss = myNetwork.compute_cost(Y_train)

        #now test....
        preds = myNetwork.predict(data_test.T)
        acc = accuracy_score(np.argmax(preds, axis=0),
                             np.argmax(Y_test.T, 1))

        train_.append([i,loss,acc])
        print("Accuracy: {:.1f}%".format(acc*100.))

        line1.set_data(
            np.array(train_)[:,0],
            np.array(train_)[:,2],
                       )
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        plt.draw()


    myNetwork.save("test.pkl")

    print("done")
    plt.show()








