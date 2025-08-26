import numpy as np


    #Implementation of a single neuron

def sigmoid(x):
    # Our activation function 
    return 1/(1+np.exp(-x))
    
class Neuron:
    
    def __init__(self,weight,bias):
        self.weight = weight
        self.bias = bias
        
    def feed_forward(self,input):
        total = np.dot(self.weight,input)+self.bias
        return sigmoid(total)


weight = np.array([0,1])
bias =4
n = Neuron(weight,bias)

x=np.array([2,3])
print(n.feed_forward(x))    





