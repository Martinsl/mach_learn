import sys
import os
from os.path import join
from skimage import feature
from skimage.feature import multiblock_lbp
import numpy as np
import cv2


# settings for LBP
METHOD = 'uniform'
radius = 3
n_points = 8 * radius


def getHistogram(img_path, img_class):
    image = cv2.imread(img_path)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = multiblock_lbp(grayImage, 0, 0, 90, 90)
    hist, _ = np.histogram(lbp.ravel(), 256, [0, 256])

    return lbp  # np.append(hist, img_class)


if len(sys.argv) < 2:
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

np_hists = np.array(hists)

# Removendo todos valores que a soma da coluna eh zero
np_hists = np_hists[:, np_hists.sum(axis=0) > 0]
np.random.shuffle(np_hists)

f = file("lbp_hist_rand", "w")

f.write("%s %s\n" % (len(np_hists), len(np_hists[0])))
for i in range(len(np_hists)):
    for j in range(len(np_hists[i]) - 1):
        f.write("%s " % np_hists[i][j])
    f.write("%s\n" % np_hists[i][-1])
