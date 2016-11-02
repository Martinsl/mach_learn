import sys
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def getArgs(argv):
    msg = 'Usage: python read_file -f <samples_file>'
    if (len(argv) != 3):
        print '\n\t', msg, '\n'
        exit()
    file = argv[argv.index('-f') + 1]

    return file


def readFile(fileName):
    mat = []
    classes = []
    with open(fileName) as fileobject:
        for line in fileobject:
            row = line.strip().split(' ')
            classes.append(int(row.pop()))
            mat.append(np.array(map(float, row)))
    row = int(mat.pop(0))
    cols = classes.pop(0)
    return mat, classes


fileName = getArgs(sys.argv)


print 'Reading files...'
allSamples, allClasses = readFile(fileName)
print 'Done.'

trSmpl, tsSmpl, trLbls, tsLbls = train_test_split(
    allSamples, allClasses, test_size=0.40, random_state=42)

clf = tree.DecisionTreeClassifier(max_depth=10)
print 'Fitting...'
clf = clf.fit(trSmpl, trLbls)

print 'Predicting...'
treeResult = clf.predict(tsSmpl)
print 'Prediction done.'

print 'Creating matConfusao.'
print confusion_matrix(tsLbls, treeResult)
print accuracy_score(tsLbls, treeResult)

with open("tree_3.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
# print "Predicted classes: ", treeResult
# print "Real classes: ", tsLbls
