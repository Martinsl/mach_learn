import sys
from get_samples import get_info
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
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

bagging = BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)
