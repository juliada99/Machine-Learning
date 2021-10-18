"""
Julia Adamczyk
Submission date: 9/23/2021
History 
9/14/21 - set up a file
9/15/21 - finished the algorithm
"""
import math
import numpy as np
import scipy.spatial.distance as distance
from random import randint

def K_Means(dataset, K, mu):
    # get rid of size 1 insignificant dimensions that got created during data preprocessing
    dataset = np.squeeze(dataset)
    # get the dimensionality of the sample
    sample_dim = dataset.ndim
    # current mu will be assigned later
    current_mu = None
    # create a variable that will hold the mu calculated in previous iteration (in first iteration it is none)
    last_mu = np.empty((K, sample_dim))
    # if mu is not provided or number of mu entries provided does not match K, or is the mu value has wrong dimensionality
    # here calculate the min and max of each of the dimension of the dataset to narrow down the choice
    maximum = np.amax(dataset, axis=0)
    minimum = np.amin(dataset, axis=0)
    if len(mu) == 0 or mu.shape[0] != K or mu.shape[1] != sample_dim:
        current_mu = np.random.randint(minimum, maximum, size=(K, sample_dim))
    else:
        current_mu = mu
    # while current mu is different than last mu
    while not np.array_equal(current_mu, last_mu):
        # store clusters
        clusters = {}
        for k in range(1, K+1):
            clusters[k] = []
        # for each point
        for point in dataset:
            # store distances
            distances = []
            # for each cluster mean 
            for cluster_mean in current_mu:
                # calculate distances 
                distances.append(distance.euclidean(cluster_mean, point))
            # pick the cluster with the smallest distance (return it's index)
            indices = [i for i, x in enumerate(distances) if x == min(distances)]
            # append the point to the nearest cluster
            clusters[indices[0]+1].append(point) 
        # recalculate the mean for each cluster
        new_mean = []
        for k in range(1, K+1):
            cluster = np.array(clusters[k])
            new_mean.append(np.mean(cluster, axis=0))
        last_mu = current_mu
        current_mu = np.array(new_mean)  
        #print("last ", last_mu)
        #print("current ", current_mu)    
    return current_mu