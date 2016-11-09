import sys
from get_samples import get_info
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def getArgs(argv):
    msg = 'Usage: python read_file -f <samples_file>'
    if (len(argv) != 3):
        print '\n\t', msg, '\n'
        exit()
    file = argv[argv.index('-f') + 1]

    return file


fileName = getArgs(sys.argv)

trSmpl, tsSmpl, trLbls, tsLbls = get_info(fileName, 0.40)

kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

clf = svm.SVC(kernel='linear')
print 'Fitting...'
clf = clf.fit(trSmpl, trLbls)

print 'Predicting...'
treeResult = clf.predict(tsSmpl)
print 'Prediction done.'

print 'Creating matConfusao.'
print confusion_matrix(tsLbls, treeResult)
print accuracy_score(tsLbls, treeResult)
