import sys
from get_samples import get_info
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def getArgs(argv):
    msg = 'Usage: python knn -f <samples_file>'
    if (len(argv) != 3):
        print '\n\t', msg, '\n'
        exit()
    file = argv[argv.index('-f') + 1]

    return file


fileName = getArgs(sys.argv)

trSmpl, tsSmpl, trLbls, tsLbls = get_info(fileName, 0.40)

neigh = KNeighborsClassifier(n_neighbors=7, weights='distance')

print 'Fitting...'
neigh.fit(trSmpl, trLbls)

print 'Predicting...'
neighResult = neigh.predict(tsSmpl)
print 'Prediction done.'

print 'Creating matConfusao.'
print confusion_matrix(tsLbls, neighResult)
print accuracy_score(tsLbls, neighResult)
