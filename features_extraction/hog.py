import sys
import os
from os.path import join
from skimage.feature import hog
import numpy as np
import cv2


def getHistogram(img_path, img_class):
    image = cv2.imread(img_path)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    or_fd = hog(grayImage, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1))
    print len(or_fd)

    return np.append(or_fd, img_class)


def normalize(arr):
    maxI = np.copy(arr[0])
    j = 0
    for i in xrange(1, len(arr)):
        for j in xrange(len(arr[i]) - 1):
            if arr[i][j] > maxI[j]:
                maxI[j] = arr[i][j]

    for i in xrange(0, len(arr)):
        for j in xrange(0, len(arr[i]) - 1):
            arr[i][j] = arr[i][j] / maxI[j]

    return arr


if len(sys.argv) < 3:
    print "\n\tusage: load_hists.py <imgs_dir_path>\n"
    exit()

imgs_path = sys.argv[1]

hists = []

for i in os.listdir(imgs_path):
    if '.' not in i:
        print i
        class_path = join(imgs_path, i)
        for j in os.listdir(class_path):
            if '.DS' not in j and 'out' not in j:
                print '\t', j
                hists.append(getHistogram(join(class_path, j), int(i)))
                exit()

# Removendo todos valores que a soma da coluna eh zero
np_hists = np.array(hists)
np_hists = np_hists[:, np_hists.sum(axis=0) > 0]

# Normalizando colunas, tomar cuidado com ultima col
np_hists = normalize(np_hists.astype(np.float64))
np.random.shuffle(np_hists)

f = file("hog", "w")

f.write("%s %s\n" % (len(np_hists), len(np_hists[0])))
for i in range(len(np_hists)):
    for j in range(len(np_hists[i]) - 1):
        f.write("%s " % np_hists[i][j])
    f.write("%s\n" % np_hists[i][-1].astype(np.int64))
