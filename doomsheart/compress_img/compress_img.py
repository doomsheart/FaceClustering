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
plt.show()
img_data = lenna / 255.0
img_data = img_data.reshape((-1, 3))
img_data[0]

def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    pixel = data[i].T
    R, G, B = pixel[0], pixel[1],pixel[2]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0,1), ylim=(0,1))
    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0,1), ylim=(0,1))
    fig.suptitle(title, size=20)
    fig.show()

plot_pixels(img_data, title='Input color space:')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS
img_data = img_data.astype(np.float32)
compactness, labels, centers = cv.kmeans(img_data, 16, None, criteria, 10, flags)
print(centers)