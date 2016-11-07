import sys
from get_samples import get_info
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def getArgs(argv):
    msg = 'Usage: python read_file -f <samples_file> -o <output_file>'
    if (len(argv) != 3):
        print '\n\t', msg, '\n'
        exit()
    file = argv[argv.index('-f') + 1]

    return file


fileName = getArgs(sys.argv)

trSmpl, tsSmpl, trLbls, tsLbls = get_info(fileName, 0.40)
n_features = len(trSmpl[0])
print 'n_features = ', n_features
clf = RandomForestClassifier(n_estimators=n_features)
print 'Fitting...'
clf = clf.fit(trSmpl, trLbls)

print 'Predicting...'
treeResult = clf.predict(tsSmpl)
print 'Prediction done.'

print 'Creating matConfusao.'
print confusion_matrix(tsLbls, treeResult)
print accuracy_score(tsLbls, treeResult)
