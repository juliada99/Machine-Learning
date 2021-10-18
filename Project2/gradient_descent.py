import numpy as np
import math

def gradient_descent(gradient, start_point, learning_rate):
        # while the magnitude of the gradient is not small enough
        while np.linalg.norm(gradient(start_point)) >= 0.0001:
                # take the start point and update the gradient according to the formula x = x - eta * gradient(x)
                start_point = start_point - learning_rate * (gradient(start_point))  
        # once while loop is done return the updated start point
        return start_point