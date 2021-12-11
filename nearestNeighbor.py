import numpy as np
import pandas as pd 
from copy import deepcopy
from datetime import datetime

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



def forwardSearch(data):
    features = []
    maxAccuracy = 0
    bestFeatures = []

    for j in range(len(data.vector[0])):
        print('Checking to add', len(features), 'th feature')
        # initially look for the most effetive feature

        accuracies = []
        for i in range(len(data.vector[0])):
            
            if i not in features:
                print('Checking feature', i)

                new = data.copy(deep=True)

                new.vector = data.vector.map(lambda x: x[features + [i]])

                newacc = accuracy(new)
                accuracies.append(newacc)

                print("Accuracy of feature(s): ", (features + [i]), " : ", newacc * 100, "%")
            else: 
                accuracies.append((-1))

        if np.max(accuracies) < maxAccuracy: 
            features.append(np.argmax(accuracies))
            print("Using feature", np.argmax(accuracies) , " is the best option at this level.")
        else: 
            features.append(np.argmax(accuracies))
            maxAccuracy = np.max(accuracies)
            bestFeatures = features
            print("Using feature", np.argmax(accuracies) , " as well improves the accuracy!")

def backwardElimination(data):
    features = {0,1,2,3,4,5,6,7,8,9}
    maxAccuracy = 0
    bestFeatures = []

    for j in range(len(data.vector[0])):
        print('Checking to remove', len(data.vector[0]) - len(features), 'th feature')

        accuracies = []
        for i in range(len(data.vector[0])):

            if i in features:
                print('Checking feature', i)

                new = data.copy(deep=True)

                temp = deepcopy(features)
                temp.remove(i)
                new.vector = data.vector.map(lambda x: x[list(temp)])

                newacc = accuracy(new)
                accuracies.append(newacc)

                print("Accuracy of feature(s): ", temp, " : ", newacc * 100, "%")
            else:
                accuracies.append(-1)

        if np.max(accuracies) < maxAccuracy: 
            features.remove(np.argmax(accuracies))
            print("Using feature", np.argmax(accuracies) , " is the best option at this level.")
        else: 
            features.remove(np.argmax(accuracies))
            maxAccuracy = np.max(accuracies)
            bestFeatures = features
            print("Removing feature", np.argmax(accuracies) , " as well improves the accuracy!")



# DATA CLEANING AND PREPROCESSING
df = pd.read_csv('./Ver_2_CS170_Fall_2021_Small_data__74.txt', header=None)

rows = []
for i in range(df.shape[0]):
    temp = (df.iloc[i][0].split())
    for j in range(len(temp)):

        # Here I am converting the scientific notation into floats
        # I stole this code from https://stackoverflow.com/questions/29849445/convert-scientific-notation-to-decimals
        temp[j] = float(temp[j])

    # I put the classifications into one column and the vectors in the other
    rows.append(pd.Series([temp[0], np.array(temp[1:])])) 

data = pd.DataFrame(rows)

# Rename the columns for readable code
data.columns = ['classification', 'vector'] 

start = datetime.now()
backwardElimination(data)
print(datetime.now()-start)