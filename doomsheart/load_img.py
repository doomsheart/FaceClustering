import cv2 as cv
import os
import numpy as np
IMG_PATH = "cluster_img/Cropped_imgs/2018-06-20-10-13-48-reallyreally/"
file_list = os.listdir(IMG_PATH)
file_list.sort()
img_list = []

# img = cv.cvtColor(cv.imread(IMG_PATH + file_list[0]), cv.COLOR_BGR2GRAY)
# print(img)
# print(img.max())
# print(cv.cvtColor(cv.imread(IMG_PATH + f), cv.COLOR_BGR2GRAY))
for f in file_list:
    img = cv.cvtColor(cv.imread(IMG_PATH + f), cv.COLOR_BGR2GRAY)
    max_ = img.max()
    img_info = [v / max_ for l in img for v in l]
    img_list.append(img_info)
# for l in img_list:
#     print(l)
print(img_list)
# print(len(img[0]))