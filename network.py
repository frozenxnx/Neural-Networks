import numpy as np


    #Implementation of a single neuron

def sigmoid(x):
    # Our activation function 
    return 1/(1+np.exp(-x))
    
def der_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

def mse_loss(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()

class NeuralNetwork:
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
     
         
        #weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        #biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()                    
   

    def feedforward(self,x):
        h1 = sigmoid(self.w1*x[0] + self.w2*x[1] + self.b1)
        h2 = sigmoid(self.w3*x[0] + self.w4*x[1] + self.b2)
        o1 = sigmoid(self.w5*h1 + self.w6*h2 + self.b3)
        return o1
    

    def train(self,data,all_y_trues):
        """

        """
    


        

        
        
data = np.array(
        [-2,-1]
        [25,6]
        [17,4]
        [-15,-6])

all_y_trues = np.array(
    [1,
    0,
    0,
    1,
    ]
    )









