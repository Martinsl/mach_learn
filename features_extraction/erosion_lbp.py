import sys
import os
from os.path import join
from skimage.feature import local_binary_pattern
import numpy as np
import cv2


def getHistogram(img_path, img_class):
    image = cv2.imread(img_path)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(grayImage, kernel, iterations=1)

    # cv2.imwrite(str(img_class) + 'erosion_or.jpg', erosion)

    grayImage = cv2.equalizeHist(grayImage)
    erosion = cv2.erode(grayImage, kernel, iterations=1)

    lbp = local_binary_pattern(erosion, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), 256, [0, 256])

    # cv2.imwrite(str(img_class) + 'erosion_eq.jpg', erosion)

    return np.append(hist, img_class)


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
    print "\n\tusage: load_hists.py <imgs_dir_path> <int(radius)>\n"
    exit()

imgs_path = sys.argv[1]
radius = int(sys.argv[2])

# settings for LBP
METHOD = 'uniform'
n_points = 8 * radius

hists = []

for i in os.listdir(imgs_path):
    if '.' not in i:
        print i
        class_path = join(imgs_path, i)
        for j in os.listdir(class_path):
            if '.DS' not in j and 'out' not in j:
                print '\t', j
                hists.append(getHistogram(join(class_path, j), int(i)))

# Removendo todos valores que a soma da coluna eh zero
np_hists = np.array(hists)
np_hists = np_hists[:, np_hists.sum(axis=0) > 0]

# Normalizando colunas, tomar cuidado com ultima col
np_hists = normalize(np_hists.astype(np.float64))
np.random.shuffle(np_hists)

lbpFileName = "er_eq_lbp_hist_" + str(radius)
f = file(lbpFileName, "w")

f.write("%s %s\n" % (len(np_hists), len(np_hists[0])))
for i in range(len(np_hists)):
    for j in range(len(np_hists[i]) - 1):
        f.write("%s " % np_hists[i][j])
    f.write("%s\n" % np_hists[i][-1].astype(np.int64))
