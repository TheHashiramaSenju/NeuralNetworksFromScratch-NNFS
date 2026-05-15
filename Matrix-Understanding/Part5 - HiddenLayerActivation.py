#to work with neural networks with non linear activation functions in hidden layers we need 2 or more layers to fit in 
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)
np.random.seed(0)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #JUST THE WEIGHT IN GAUSSIAN DISTRIBUTION (-1 TO 1)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass
    
#now here we wull try to build an activation ReLU function as we have seen !

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) #because ReLU is selecting activation points at max(0, i) telling where to be activated


layer1 = Layer_Dense(2, 5) # we created this initial layer
#initially we will do for 1 layer then we can maximize stuffs from thereon
#the dataset yielded has x and y representing them so 2 unique features....and ultimately 
#increasing features increases the dimensions also (Remember about LLM and ALA interconnected example)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)#this is like activation after weights and biases here 
print(activation1.output)