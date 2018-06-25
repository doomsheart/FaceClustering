import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
curr_dir = os.getcwd()
# %matplotlib inline
lenna = cv.imread(curr_dir + "\\Lenna.png", cv.IMREAD_COLOR)
plt.style.use("ggplot")
plt.rc("axes", **{"grid": False})
plt.imshow(cv.cvtColor(lenna, cv.COLOR_BGR2RGB))