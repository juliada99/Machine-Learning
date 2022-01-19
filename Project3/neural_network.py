import matplotlib as plt
import numpy as np
from numpy.core.fromnumeric import argmax
import sklearn


'''
calculates the loss given current model, samples and correct labels
'''
def calculate_loss(model, X, y):
        # a matrix is nn_hdim x nn_hdim matrix 
        a = X.dot(model['W1']) + model['b1']
        # h matrix is nn_hdim x nn_hdim matrix 
        h = np.tanh(a)
        # z matrix is nn_hdim x 2 matrix 
        z = h.dot(model['W2']) + model['b2']
        # prediction is (nn_input, 2) y is (nn_input, )
        prediction = softmax(z)
        # transform y arrayy into (nn_input, 2) to enable multiplication
        y_true = np.asarray([[1-v, v] for v in y])  
        loss = -np.sum(y_true*np.log(prediction))
        return loss * 1.0/len(X)

def predict(model, x):
        # compute hidden layer
        a = x.dot(model['W1']) + model['b1']
        # apply non-linearity to hidden layer
        h = np.tanh(a)
        # compute output layer
        z = h.dot(model['W2']) + model['b2']
        # apply non-linearity to output
        prediction = softmax(z)
        # return the maximum of predictions' probabilities for each sample
        return np.argmax(prediction, axis=1)
'''
returns a dictionary with weights W1, W2, and biases b1, b2
'''
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
        # initialize learning rate
        learning_rate = 0.01
        # seed
        np.random.seed(0)
        # initialize weights 
        W1 = np.random.uniform(0.0, 1.0, (X.shape[1], nn_hdim))
        b1 = np.random.uniform(0.0, 1.0, (nn_hdim))
        W2 = np.random.uniform(0.0, 1.0, (nn_hdim, 2))
        b2 = np.random.uniform(0.0, 1.0, (2))
        # initialize the model
        model = {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}
        # loop through the number of epochs
        for i in range(0, num_passes):
                # forward propagation calculations
                a = X.dot(model['W1']) + model['b1']
                h = np.tanh(a)
                z = h.dot(model['W2']) + model['b2']
                prediction = softmax(z)
                # fit y for broadcasting where y = 0 y_out = [1,0], y = 1 y_out = [0,1] 
                y_out = np.asarray([[1-v, v] for v in y])  

                # backpropagation calculations
                dL_dy = np.subtract(prediction, y_out)      # shape (n_samples, n_output_dim)
                helper = dL_dy.dot(model['W2'].T)       # should be shape (200, nn_hid_dims)
                dL_da = (1 - np.power(h, 2)) * helper   # shape (n_samples, n_hidden dimensions) 
                dL_dW2 = (h.T).dot(dL_dy)       # shape (nn_hid_dims, 2)
                dL_db2 = np.sum(dL_dy, axis=0)  # use sum to get rid of dimension
                dL_dW1 = (X.T).dot(dL_da) # should be shape (n_inputs, nn_hid_dims)
                dL_db1 = np.sum(dL_da, axis=0) # use sum to get rid of dimension

                # update weights and biases
                W1 -= learning_rate * dL_dW1
                b1 -= learning_rate * dL_db1
                W2 -= learning_rate * dL_dW2
                b2 -= learning_rate * dL_db2

                # update the model
                model = {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}

                # print loss
                if print_loss and i % 1000 == 0:
                        print(calculate_loss(model, X, y))
        # return the model
        return model
                
def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
