import numpy as np

class Perceptron(object):


    def __init__(self,bias=0,threshold=0,learning_rate=0.01):
        self.bias=bias
        self.learning_rate=learning_rate
        self.threshold=threshold 

    def sgd(self):
        print("Stochastic Gradient Descent")

    def train(self):
        print("TRAINING....")


    def sigmoid(self):
        print("SIGMOID")

# Inputs
X_train=np.array([
    [1,1],
    [1,0],
    [0,1],
    [0,0],
])

y=np.array([1,0,0,0])




p=Perceptron()

