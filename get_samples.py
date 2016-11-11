import numpy as np
import random
from datetime import datetime
from sklearn.model_selection import train_test_split


def readFile(fileName):
    mat = []
    classes = []
    with open(fileName) as fileobject:
        for line in fileobject:
            row = line.strip().split(' ')
            classes.append(int(row.pop()))
            mat.append(np.array(map(float, row)))
    row = int(mat.pop(0))
    classes.pop(0)  # extracts number of cols, col = classes.pop(0)

    return mat, classes


def get_info(filePath, trPercent):
    rndState = random.seed(datetime.now())
    allSamples, allClasses = readFile(filePath)

    return train_test_split(
        allSamples, allClasses, test_size=trPercent, random_state=rndState)
