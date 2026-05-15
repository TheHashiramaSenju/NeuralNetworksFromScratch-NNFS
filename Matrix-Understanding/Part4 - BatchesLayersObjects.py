#learning rates and batch sizes are pretty much closely related to each other 
#now we will know how to batch the inputs and outputs

import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]


weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.9, 0.26, -0.5],
           [-0.26, 0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]


weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]


#this transpose is just an mathematical trick here to make things work here
#can also be called as a computational convinience so that through transpose everything goes on perfectly through all the nodes
#since this is actually an array we are doing a proper matrix multiplication instead of the series X the matrix sequentially

layer1_output1 = np.dot(inputs, np.array(weights).T) + biases
print(layer1_output1)

# now we will do for another layer, so as natuarlly exectued in a neural network. 
#the next layer uses the previous layer's computed outputs and process it with its own weight and biases 
#which naturally results in 

layer2_output2 = np.dot(layer1_output1, np.array(weights2).T) + biases2
print(layer2_output2)



'''Now we will do some magic here'''


X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #JUST THE WEIGHT IN GAUSSIAN DISTRIBUTION (-1 TO 1)
        self.biases = np.zeros((1, n_neurons))
        
        '''
        we when deciding to build a neural network must be aware of the inputs and the desired outputs that comes in and out of the neural network
        in order to properly initialize the weights in a proper matrix format
        in general - number of inputs  -> number of the features (columns) present (think intuitively here)
        '''
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass
    
    
layer1 = Layer_Dense(4, 5) # we created this initial layer
layer2 = Layer_Dense(5, 2) # we created this another layer getting a forward pass kind of

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)



#we want small values in neural networks because of the explosion that might cause in propagations
#scaling helps - meaning stays the same but we just do some mathematical stuffs in order to keep it in a proper and a safe limit
#initialize biases other than 0 because if it is not done the neuron may not fire and the same 0 stuffs gets propagated and ultimately resulting in a non-working neural network 


'''
saving and loading a model essentially means that we are actually 
saving weights and biases and refering them in latter to work with them 
'''