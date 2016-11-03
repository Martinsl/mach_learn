import sys
from get_samples import get_info
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def getArgs(argv):
    msg = 'Usage: python read_file -f <samples_file>'
    if (len(argv) != 3):
        print '\n\t', msg, '\n'
        exit()
    file = argv[argv.index('-f') + 1]

    return file


def fitness(trSmpl, tsSmpl, trLbls, tsLbls, chromosome):
    fit = 0.0

    return fit


fileName = getArgs(sys.argv)

trSmpl, tsSmpl, trLbls, tsLbls = get_info(fileName, 0.40)

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
