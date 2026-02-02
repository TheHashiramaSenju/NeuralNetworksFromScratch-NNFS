#Here we will see a very simple implementation of a neuron (from matrix) to initially understand and take it forward
#Please refer to the each part's Notes.md for understanding the code and the neuron implementations

inputs = [2.0, 4.1, 3.0]
weight = [0.1, 0.21, 0.41]
bias = 2.0
#the equation translates as y = m*x + c where c is the bias (Innate happiness of the neuron)
y_out = (inputs[0]*weight[0] + inputs[1] * weight[1] + inputs[2] * weight[2]) + bias 
print(y_out)
