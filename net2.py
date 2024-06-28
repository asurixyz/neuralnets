import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

def mse_loss(y_true,y_pred):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork :
    '''
    We have a neural network with 
    - 2 inputs (x1,x2)
    - 6 weights (w1,w2,w3,w4,w5,w6)
    - 3 biases (b1,b2,b3)

    Three layered - one input layer, a hidden layer with 2 neurons (h1,h2)
    and an output layer with one neuron (o1)
    '''

    def __init__(self):
        #Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        #Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self,x):
        h1 = sigmoid(self.w1*x[0] + self.w2*x[1] + self.b1)
        h2 = sigmoid(self.w3*x[0] + self.w4*x[1] + self.b2)
        o1 = sigmoid(self.w5*h1 + self.w6*h2 + b3)
        return o1
    
    def train(self,data,all_y_true):
        '''
        - data is an (n x 2) numpy array
        - all_y_true is an (n x 1) numpy array
        '''
        learn_rate = 0.1
        epochs = 1000 #Number of times to loop through the dataset

        for epoch in range(epochs):
            for x,y_true in zip(data,all_y_true):
                sum_h1 = self.w1*x[0] + self.w2*x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3*x[0] + self.w4*x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5*h1 + self.w6*h2 + self.b3 
                o1 = sigmoid(sum_o1)
                y_pred = o1

                #Using Chain Rule
                dL_dypred = -2*(y_true - y_pred)
                dypred_dh1 = self.w5*deriv_sigmoid(self.w5 * h1 + self.w6*h2 + self.b3)

    
