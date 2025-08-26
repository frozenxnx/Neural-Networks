import numpy as np


    #Implementation of a single neuron

def sigmoid(x):
    # Our activation function 
    return 1/(1+np.exp(-x))
    
class Neuron:
    
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias
        
    def feed_forward(self,inputs):
        total = np.dot(self.weights,inputs)+self.bias
        return sigmoid(total)



class NeuralNetwork:

#assume all neurons have the same  weights w= [0,1],  the same bias b=0 and 
# the same sigmoid activation function. Let h1,h2,o1 denote the outputs of the neurons they represent.
    '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
    '''     
    def __init__(self):
        weights = np.array([0,1])
        bias =0

        self.h1 = Neuron(weights,bias)
        self.h2 = Neuron(weights,bias)
        self.o1 = Neuron(weights,bias)

    def feedforward(self,x):
        out_h1 = self.h1.feed_forward(x)
        out_h2 = self.h2.feed_forward(x)

        out_o1 = self.o1.feed_forward(np.array([out_h1,out_h2]))
        return out_o1
        
network = NeuralNetwork()
x= np.array([2,3])
print(network.feedforward(x))








