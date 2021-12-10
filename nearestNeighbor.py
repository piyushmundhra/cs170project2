import pandas as pd 
import numpy as np 

# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def nearestNeighbor(idx, data):
    closest = None
    for i in range(data.shape[0]):

        # This is basically the base case, it had to be hardcoded to prevent KeyErrors
        if closest == None and i != idx: closest = i

        # Make sure we don't keep marking the closest as itself
        elif i != idx:

            # If the current vector is closer than "closest", then update closest
            if np.linalg.norm(data.vector[i] - data.vector[idx]) < np.linalg.norm(data.vector[closest] - data.vector[idx]): closest = i

    # Return the classification of "closest"
    return data.classification[closest]

def accuracy(data):

    # Running total of correct predictions
    correctPredictions = 0

    for i in range(data.shape[0]):

        # if the closet neighbor has the same classification as the god given classification, then increment "correcPredictions" by 1
        if nearestNeighbor(i, data) == data.classification[i]: correctPredictions += 1
    
    # Return the total number of correct predictions divided by the total number of pieces of data
    return correctPredictions / data.shape[0]

def search():
    pass