import numpy as np


def tubify(img, d):

    img3d = []
    for k in range(d):
        mu = np.exp(-(1/d)*(k-(d/2)) ** 2)
        img3d.append(img * mu)

    return np.array(img3d)


def get_img(img, idx):
    return np.reshape(img[idx, :], (28, 28))
