#here we are gonna add one more input layer
#before we actually had 3 inputs in part 1 so now we will add one more which actuallu is

#input layer --> The values that we are actually tracking, A value from an sensor

#here we will try to model the output og a neuron for a change -->takes input from the hidden layer

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 10] #Each line corresponds to the weights (refer the the notes material)
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias 
print(output)


'''
Now we are going to add more neurons kinda.
think of it like this --> 3 NEURONS WITH 4 INPUTS EACH! (Surprising Isn't it ?)
check in the notes for this.

So now for each neuron of those 3 neurons they have 4 different biases totally and again 
4 weights for each of them. So this roughly translates this to
'''


#each circle (neuron) has the following inputs, 3 bias in total and 4 weights each + the common input values (refer to the respective notes part )
inputs = [1, 2, 3, 2.5]
weights1, bias1 = [0.2, 0.8, -0.5, 1.0], 2
weights2, bias2 = [0.5, -0.91, 0.26, -0.5], 3
weights3, bias3 = [-0.26, -0.27, 0.17, 0.07 ], 0.5

output2 = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + bias1,
          inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + bias2,
          inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + bias3
          ] 

print(output2)



#diving deeper - Transforming lists into matrices and using loops

inputs = [1, 2, 3, 2.5]
weights1 = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5], 
            [-0.26, -0.27, 0.17, 0.07 ]]
biases = [2, 3, 0.5]


print(zip(weights1, inputs))  #zip function pairs elements from two lists into tuples

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights1, biases): #outer row iteratin
    neuron_output = 0 
    for n_input, weight in zip(inputs, neuron_weights): #inner row iterations # since i and j come together they go like 1,1 2,2 3,3 WOAHHH 
        neuron_output = neuron_output + (n_input * weight)
    neuron_output = neuron_output + neuron_bias
    layer_outputs.append(neuron_output)
    
print(layer_outputs)



#weights and biases are the optimizers that actually tune to make the neural network learn.

some_value = -0.5
weight = 0.7
bias = 0.7

print(some_value)


#we will deal with activation functions that helps us understand more about this 

'''...rest continued on chapter 3 and refer the notes for more further informations'''