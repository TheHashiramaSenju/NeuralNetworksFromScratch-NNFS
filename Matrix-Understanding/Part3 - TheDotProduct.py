
'''
Now we will deal with the shapes in neural network
At each dimensiton what is the size of that dimension
'''

# we need HOMOLOGOUS shape like an array does (3 for 3 arrays kind of we need it) 
#tensor is an object that can be repersented as an array (in machine learning and Neural network concept)

'''*RECAP*'''
import numpy as np
inputs2 = [1, 2, 3, 2.5] #this is a vector 

weights2 = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5], 
            [-0.26, -0.27, 0.17, 0.07 ]] #3 neurons

biases2 = [2, 3, 0.5] #this is a vector too 

#we use dot product now --> can be seen visually in 3blue 1brown
#all dotted product lead to a scalar single value

'''START: Now we will start using numpy'''
#it seems pretty simple in numpy that we did as individual loops before 
outputs = np.dot(weights2, inputs2) + biases2 #here the first vector denotes what gets indexed ! 'refer to the notes
print(outputs)

'''refer to the notes part for a detailed explanation'''

