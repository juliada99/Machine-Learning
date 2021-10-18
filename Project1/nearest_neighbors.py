"""
Julia Adamczyk
Submission date: 9/23/2021
History 
9/9/21 - set up a file
9/10/21 - wrote logic of the KNN_test
9/11/21 - finished up KNN_test
"""
import numpy as np
import scipy.spatial.distance as distance

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    # for each point in the test set save the predicted label
    predicted_labels = []
    for i in X_test:
        # save distances to each of the points
        distances = []
        # for each point in the training set
        for j in X_train:
            # collect distances
            distances.append(distance.euclidean(i, j))
        # zip into tuple    
        distance_label_tuple = list(zip(distances, Y_train))
        # sort based on the distance key
        distance_label_tuple.sort()
        # sum the labels from K nearest neighbors 
        sum = 0
        for k in range(K):
            sum += distance_label_tuple[k][1]
        # if sum is greater than 0 predict 1, else predict -1
        if sum > 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(-1)
    # count correct labels
    correct_labels = 0
    for a, b in zip(Y_test, predicted_labels):
        if a == b:
            correct_labels += 1
    # get total labels
    total_labels = len(Y_test)
    # return the ratio: correct/total 
    acc = float(correct_labels/total_labels)
    return acc
