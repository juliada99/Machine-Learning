import numpy as np
import math

def perceptron_train(X,Y):
        # get the dimensionality of the data
        n_dims = X.shape[1]
        n_samples = X.shape[0]
        # initialize bias to 0
        bias = 0
        # initialize weights to 0
        weights = np.zeros(n_dims, dtype=float)
        # go through samples
        # repeat till convergence (all samples classified correctly) or set number of epochs
        # Y - correct labels
        # C - current labels (let's initialize them to 0)
        C = np.zeros(n_samples, dtype=int)
        # while not all samples classified correctly
        while not (Y==C).all():
                # basically do one epoch (loop through the whole set)
                for index, sample in enumerate(X):
                        # calculate activation and predict label to update current labels
                        activation, pred_lbl = compute_activation_and_label(sample, weights, bias)
                        # update current labels
                        C[index] = pred_lbl
                        # check update condition (ya <= 0)
                        if Y[index]*activation <= 0:
                                # update weights and bias
                                weights, bias = update_weights_and_bias(sample, Y[index], weights, bias)
        return weights, bias

def perceptron_test(X_test, Y_test, w, b):
        # store correctly predicted labels
        correct_labels = 0
        # for each sample in test set
        for index, sample in enumerate(X_test):
                # predict label given weights and bias
                _, lbl = compute_activation_and_label(sample, w, b)
                # if predicted label is the same as the test label
                if lbl == Y_test[index]:
                        # increase the number of correctly predicted labels
                        correct_labels += 1
        # return the ratio 
        return float(correct_labels/Y_test.shape[0])    

def compute_activation_and_label(sample, weights, bias):
        # activation is a dot product of a sample and weights plus bias
        activation = np.dot(sample, weights) + bias
        # predict the label based on activation
        predicted_label = None
        if activation > 0:
                predicted_label = 1
        else:
                predicted_label = -1
        # return: activation, label
        return activation, predicted_label

def update_weights_and_bias(sample, label, weights, bias):
        # update bias and weights according to update rules
        new_bias = label + bias
        new_weights = weights + label * sample
        return new_weights, new_bias 