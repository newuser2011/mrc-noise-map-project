import numpy as np
import matplotlib.pyplot as plt


class Model:

    def __init__(self, input_shape, hidden1_size, hidden2_size, hidden3_size, hidden4_size,  hidden5_size):  #, hidden6_size):  #, hidden7_size, hidden8_size):
        self.input_size = input_shape
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.hidden4_size = hidden4_size
        self.hidden5_size = hidden5_size
        self.output_size = 1
        self.w1 = np.random.randn(self.input_size, self.hidden1_size) / np.sqrt(self.hidden1_size) #xavier initialisation
        self.w2 = np.random.randn(self.hidden1_size, self.hidden2_size) / np.sqrt(self.hidden2_size)
        self.w3 = np.random.randn(self.hidden2_size, self.hidden3_size) / np.sqrt(self.hidden3_size)
        self.w4 = np.random.randn(self.hidden3_size, self.hidden4_size) / np.sqrt(self.hidden4_size)
        self.w5 = np.random.randn(self.hidden4_size, self.output_size) / np.sqrt(self.output_size)
        self.w6 = np.random.randn(self.hidden5_size, self.output_size) / np.sqrt(self.output_size)
        self.b1 = np.zeros((1, self.hidden1_size))
        self.b2 = np.zeros((1, self.hidden2_size))
        self.b3 = np.zeros((1, self.hidden3_size))
        self.b4 = np.zeros((1, self.hidden4_size))
        self.b5 = np.zeros((1, self.output_size))
        self.b6 = np.zeros((1, self.output_size))

    def forward(self, data, p):
        '''
        forward propogation through the model using relu non linearity and a dropout
        '''
        self.z1 = np.dot(data, self.w1) + self.b1
        self.z1[self.z1 < 0] = 0 #the relu non linearity
        self.u1 = np.random.binomial(1, p, size=self.z1.shape) / p
        self.z1 *= self.u1
        self.z2 = np.dot(self.z1, self.w2) + self.b2
        self.z2[self.z2 < 0] = 0
        self.u2 = np.random.binomial(1, p, size=self.z2.shape) / p
        self.z2 *= self.u2
        self.z3 = np.dot(self.z2, self.w3) + self.b3
        self.z3[self.z3 < 0] = 0
        self.u3 = np.random.binomial(1, p, size=self.z3.shape) / p
        self.z3 *= self.u3
        self.z4 = np.dot(self.z3, self.w4) + self.b4
        self.z4[self.z4 < 0] = 0
        self.u4 = np.random.binomial(1, p, size=self.z4.shape) / p
        self.z4 *= self.u4
        self.z5 = np.dot(self.z4, self.w5) + self.b5
    

    def compute_loss(self, labels):
        '''
        using MSE loss
        '''
        self.loss = np.mean((self.z5 - labels)**2)/2


    def set_weights(self, weights):
        '''
        sets the weights of the model
        '''
        self.w1 = weights[0]
        self.b1 = weights[1]
        self.w2 = weights[2]
        self.b2 = weights[3]
        self.w3 = weights[4]
        self.b3 = weights[5]
        self.w4 = weights[6]
        self.b4 = weights[7]
        self.w5 = weights[8]
        self.b5 = weights[9]

    def backward(self, data, labels):
        
        n = len(data)
        dz5 = 1/n * (self.z5- labels)

        self.dw5 = 1/n * np.dot(self.z4.T, dz5)
        self.db5 = 1/n * np.sum(dz5)
        dz4 = np.dot(dz5, self.w5.T)
        dz4[self.z4 <= 0] = 0
        dz4 *= self.u4
        self.dw4 = 1/n * np.dot(self.z3.T, dz4)
        self.db4 = 1/n * np.sum(dz4)
        dz3 = np.dot(dz4, self.w4.T)
        dz3[self.z3 <= 0] = 0
        dz3 *= self.u3
        self.dw3 = 1/n * np.dot(self.z2.T, dz3)
        self.db3 = 1/n * np.sum(dz3)
        dz2 = np.dot(dz3, self.w3.T)
        dz2[self.z2 <= 0] = 0
        dz2 *= self.u2
        self.dw2 = 1/n * np.dot(self.z1.T, dz2)
        self.db2 = 1/n * np.sum(dz2)
        dz1 = np.dot(dz2, self.w2.T)
        dz1[self.z1 <= 0] = 0
        dz1 *= self.u1
        self.dw1 = 1/n * np.dot(data.T, dz1)
        self.db1 = 1/n * np.sum(dz1)
        self.dw5 += self.l2_reg * self.w5
        self.dw4 += self.l2_reg * self.w4
        self.dw3 += self.l2_reg * self.w3
        self.dw2 += self.l2_reg * self.w2
        self.dw1 += self.l2_reg * self.w1


    def sgd_update(self, lr):
        '''
        Updating the weights using gradient descent
        '''
        self.w1 -= lr * self.dw1
        self.w2 -= lr * self.dw2
        self.w3 -= lr * self.dw3
        self.w4 -= lr * self.dw4
        self.w5 -= lr * self.dw5
        #self.w6 -= lr * self.dw6
        #self.w7 -= lr * self.dw7
        #elf.w8 -= lr * self.dw8
        #self.w9 -= lr * self.dw9
        self.b1 -= lr * self.db1
        self.b2 -= lr * self.db2
        self.b3 -= lr * self.db3
        self.b4 -= lr * self.db4
        self.b5 -= lr * self.db5
        #self.b6 -= lr * self.db6
        #self.b7 -= lr * self.db7
        #self.b8 -= lr * self.db8
        #self.b9 -= lr * self.db9

    def save(self, path):
        np.save(path + "/w1.npy", self.w1)
        np.save(path + "/w2.npy", self.w2)
        np.save(path + "/w3.npy", self.w3)
        np.save(path + "/w4.npy", self.w4)
        np.save(path + "/w5.npy", self.w5)
        #np.save(path + "/w6.npy", self.w6)
        #np.save(path + "/w7.npy", self.w7)
        np.save(path + "/b1.npy", self.b1)
        np.save(path + "/b2.npy", self.b2)
        np.save(path + "/b3.npy", self.b3)
        np.save(path + "/b4.npy", self.b4)
        np.save(path + "/b5.npy", self.b5)
        #np.save(path + "/b6.npy", self.b6)
        #np.save(path + "/b7.npy", self.b7)

    def load(self, path):
        self.w1 = np.load(path + "/w1.npy")
        self.w2 = np.load(path + "/w2.npy")
        self.w3 = np.load(path + "/w3.npy")
        self.w4 = np.load(path + "/w4.npy")
        self.w5 = np.load(path + "/w5.npy")
        #self.w6 = np.load(path + "/w6.npy")
        #self.w7 = np.load(path + "/w7.npy")
        self.b1 = np.load(path + "/b1.npy")
        self.b2 = np.load(path + "/b2.npy")
        self.b3 = np.load(path + "/b3.npy")
        self.b4 = np.load(path + "/b4.npy")
        self.b5 = np.load(path + "/b5.npy")
        #self.b6 = np.load(path + "/b6.npy")
        #self.b7 = np.load(path + "/b7.npy")









'''This code defines a neural network model with multiple hidden layers, which uses the ReLU activation
 function and dropout regularization during training. The model is implemented using NumPy and Matplotlib 
 libraries.

The constructor of the model takes several arguments, including the shape of the input data, the size of
 each hidden layer, and the output size. The weights and biases for each layer are initialized randomly 
 using the Xavier initialization method, and the forward function applies the ReLU activation function and
  dropout regularization to the outputs of each layer.

The compute_loss function calculates the loss of the model using the mean squared error (MSE) loss 
function. The backward function performs backpropagation through the model to compute the gradients of 
the loss with respect to the weights and biases, which are used to update the parameters during training.

The commented out code suggests that the model was originally designed to have more hidden layers, 
but they were later removed. This could be due to the model being too complex or difficult to train
 with additional layers, or due to the data not requiring a deeper architecture.




'''