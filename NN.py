import numpy as np
from Layer import Layer
import math

"""
    NN: is a simple neural network model for classification & regression problems
    ....

    Attributes
    ----------
        X:  type -> np.ndarray
            the input data
        Y: type -> np.ndarray
            the target data
        output_activation: type-> string
            the activation function of the last layer,
            the output layer
            default -> 'sigmoid'
    
    Example
    -------
    > from NN import nn
    > from Layer import Layer
    > # import some data from sklearn library
    > from sklearn.datasets import load_breast_cancer
    > inputs = data.data
    > targets = data.target.reshape(-1,1)
    > neural_network_model = nn(inputs, targets)
    > # add hidden layers
    > neural_network_model.add_layer( Layer(32, activation='relu') )
    > neural_network_model.fit()
    > # predict data
    > Y_pred = neural_network_model.predict(INPUTS)
    > # plot cost function
    > import matplotlib.pyplot as plt
    > plt.plot(neural_network_model._costs)
    > plt.show()

"""


class NN:

    def __init__(self, X, Y, output_activation='sigmoid'):
        self._X = X
        self._Y = Y
        self._layers = []
        self._output_activation = output_activation

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise Exception("Invalid Type", type(layer), " != <class 'Layer'>")
        self._layers.append(layer)

    # Train data

    def fit(self, learning_rate=0.01, iteration=1000):
        self._setup()
        self._costs = []
        self._learning_rate = learning_rate
        self._iteration = iteration

        # Adam parameter
        # VdW = 0
        # Vdb = 0
        # SdW = 0
        # Sdb = 0
        # beta1 = 0.9
        # beta2 = 0.999
        # epsilon = math.pow(10, -8)

        for i in range(iteration):
            self._fowardPropagation()
            self._backPropagation()

            # Adam implementation
            # self._fowardPropagation()
            # self._backPropagation_Adam(
            #     VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, i)

            print(self._calc_cost(self._layers[len(self._layers)-1].values))
            if(i % 100 == 0):
                self._costs.append(self._calc_cost(
                    self._layers[len(self._layers)-1].values))
        path_weights = "Pre-Weights/weight"
        path_biases = "Pre-Weights/bias"
        for j in range(len(self._layers)):
            np.savetxt(path_weights + str(j) + ".csv",
                       self._layers[j].weight, delimiter=",")
            np.savetxt(path_biases + str(j) + ".csv",
                       self._layers[j].bias, delimiter=",")

    # return the cost function

    def _calc_cost(self, Y_pred):
        # return np.sum(np.square(self._Y - Y_pred) / 2)

        # no regularization
        # return (-1/self._X.shape[0]) * np.sum(self._Y.T @ (np.log(Y_pred)) + (1 - self._Y).T @ (np.log(1 - Y_pred)))

        # adding regularization
        regu_part = 0
        for layer in self._layers:
            regu_part += np.sum(np.power(layer.weight, 2))
        regu_para = 0.1
        regu_part = (regu_para/(2*self._X.shape[0])) * regu_part
        return (-1/self._X.shape[0]) * np.sum(self._Y.T @ (np.log(Y_pred)) + (1 - self._Y).T @ (np.log(1 - Y_pred))) + regu_part

    # configuration the shape,
    # weight and bias of each layer
    # add output layer
    def _setup(self):
        for index, layer in enumerate(self._layers):
            if(index == 0):  # first hidden layer
                layer._setup(self._X)
            else:
                layer._setup(self._layers[index-1])
        # setup and add output layer
        output_layer = Layer(
            self._Y.shape[1], activation=self._output_activation)
        output_layer._setup(self._layers[len(self._layers)-1])
        self.add_layer(output_layer)

    def _fowardPropagation(self):
        for index, layer in enumerate(self._layers):
            if(index == 0):  # first hidden layer
                layer._foward(self._X)
            else:
                layer._foward(self._layers[index-1])

    def _backPropagation(self):
        # delta = self._Y - self._layers[len(self._layers)-1].values
        delta = (-self._Y / self._layers[len(self._layers) - 1].values) + (
            (1-self._Y) / (1 - self._layers[len(self._layers) - 1].values))
        for i in range(len(self._layers)-1, -1, -1):
            if (i == 0):  # first hidden layer
                delta = self._layers[i]._backward(
                    delta, self._X, self._learning_rate)
            else:
                delta = self._layers[i]._backward(
                    delta, self._layers[i-1], self._learning_rate)

    # backprop Adam

    def _backPropagation_Adam(self, VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, iter_num):
        # delta = self._Y - self._layers[len(self._layers)-1].values
        delta = (-self._Y / self._layers[len(self._layers) - 1].values) + (
            (1-self._Y) / (1 - self._layers[len(self._layers) - 1].values))
        for i in range(len(self._layers)-1, -1, -1):
            if (i == 0):  # first hidden layer
                delta = self._layers[i]._backward_Adam(
                    delta, self._X, self._learning_rate, VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, iter_num)
            else:
                delta = self._layers[i]._backward_Adam(
                    delta, self._layers[i-1], self._learning_rate, VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, iter_num)

    def predict(self, X_test):
        for index, layer in enumerate(self._layers):
            if(index == 0):
                layer._foward(X_test)
            else:
                layer._foward(self._layers[index-1])
        if self._is_continues():  # if target labels is continues
            return self._layers[len(self._layers)-1].values
        if self._is_multiclass():  # if target labels is multiclass
            return self._threshold_multiclass(self._layers[len(self._layers)-1])
        # binary classification
        return self._threshold(self._layers[len(self._layers)-1], 0.5)

    # set the 'predict.value' > 'value' [treshhold] to '1' others to '0'

    def _threshold(self, target, value):
        predict = target.values
        predict[predict < value] = 0
        predict[predict >= value] = 1
        return predict

    # set the max 'predict.value' to '1' others to '0'
    def _threshold_multiclass(self, target):
        predict = target.values
        predict = np.where(predict == np.max(
            predict, keepdims=True, axis=1), 1, 0)
        # predict[] = 1 | 0
        return predict

    # check if it's a multiclassfication problem
    def _is_multiclass(self):
        return len(np.unique(self._Y)) > 2

    # check if it's a regression problem
    def _is_continues(self):
        return len(np.unique(self._Y)) > (self._Y.shape[0] / 3)

    # setup pretrained weights
    def _setup_pretrained_weights(self):
        path_weights = "Pre-Weights/weight"
        path_biases = "Pre-Weights/bias"
        for index in range(len(self._layers)):
            temp1 = np.genfromtxt(
                path_biases + str(index) + ".csv", delimiter=",")
            self._layers[index].bias = np.array([temp1]).reshape(
                1, self._layers[index].shape[1])
            # output layer
            if index == 0:
                if (self._X.shape[1] == 1 or self._layers[0].shape[1] == 1):
                    temp = np.genfromtxt(
                        path_weights + str(index) + ".csv", delimiter=",")
                    self._layers[index].weight = np.array([temp]).reshape(
                        self._layers[index - 1].shape[1], self._layers[index].shape[1])
                else:
                    self._layers[index].weight = np.genfromtxt(
                        path_weights + str(index) + ".csv", delimiter=",")
            else:
                if ((self._layers[index - 1].shape[1] == 1) or (self._layers[index].shape[1] == 1)):
                    temp = np.genfromtxt(
                        path_weights + str(index) + ".csv", delimiter=",")
                    self._layers[index].weight = np.array([temp]).reshape(
                        self._layers[index - 1].shape[1], self._layers[index].shape[1])
                else:
                    self._layers[index].weight = np.genfromtxt(
                        path_weights + str(index) + ".csv", delimiter=",")

    # predict with pretrained weights

    def predict_pretrained_weights(self, X_test):
        self._setup()
        self._setup_pretrained_weights()
        return self.predict(X_test)
