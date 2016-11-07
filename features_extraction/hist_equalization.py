import sys
import os
from os.path import join
from skimage.exposure import equalize_hist
import cv2
import matplotlib.pyplot as plt


def getHistogram(img_path, img_class):
    image = cv2.imread(img_path)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage2 = cv2.equalizeHist(grayImage)

    print os.getcwd()
    filePath = os.getcwd()
    cv2.imwrite((filePath + str(img_class) + "_or.png"), grayImage)
    cv2.imwrite((filePath + str(img_class) + "_eq.png"), grayImage2)

    return grayImage2


if len(sys.argv) < 2:
    print "\n\tusage: load_hists.py <imgs_dir_path>\n"
    exit()

imgs_path = sys.argv[1]
hists = []

for i in os.listdir(imgs_path):
    if '.' not in i:
        class_path = join(imgs_path, i)
        for j in os.listdir(class_path):
            if '.DS' not in j and 'out' not in j:
                hists.append(getHistogram(join(class_path, j), int(i)))
                break
