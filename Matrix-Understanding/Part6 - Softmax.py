import math
import numpy as np 
import nnfs

nnfs.init()

layer_outputs = [4.8, 1.21, 2.385]

layer_outputs2 = [4.8, 4.79, 4.25]

#neuron exclusivity abd bounding stays a major hurdle here
'''
Here we will try using softmax functions
'''

#the problem is ReLU is that the so called function does not know how much to punish a input data.
#so we are gonna try exponentiation here (used in softmax function)

E = math.e
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)
    

print(exp_values)

#normalisation - gives the probability distribution in such a way that the offset values gets punished fairly
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print(norm_values)
print(sum(norm_values))

'''
We exponentiate and actually normalize the value, the exponentiation function
rewards and punishes the input values and all of them fairly 

and 

Normalisation function actually puts all of them in a single bar essentially shrinkin them
between 0 to 1 which all later would add up to 1 (Probability distributions)
'''
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

class Activation_Softmax:
    def forward (self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np. sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense (2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense (3, 3)
activation2 = Activation_Softmax()
dense1.forward (X)
activation1.forward(dense1.output) I
dense2. forward(activation1.output)
activation2. forward (dense2.output)
