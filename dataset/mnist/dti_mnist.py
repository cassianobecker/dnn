import numpy as np
from torchvision import datasets, transforms
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import scipy.io as sio


def batch_tensor_to_3dti(data, lift_dim=6):

    dts = []

    for k in range(data.shape[0]):

        img = data[k, 0, :, :].numpy()

        dti_img = img_to_3dti(img, lift_dim)

        dts.append(dti_img)

    tensor_dti_img = torch.tensor(dts, dtype=torch.float32)

    return tensor_dti_img


def img_to_3dti(img, lift_dim):

    dim = img.shape[0]
    w = 0.3

    dt = np.zeros([3, 3, dim, dim, lift_dim])

    for k in range(lift_dim):

        mu = np.exp(-(1 / lift_dim) * (k - (lift_dim / 2)) ** 2)
        # set minimum to zero
        imgd = img - np.min(img)
        imgd = mu * imgd
        # get the least
        z0 = np.sort(np.unique(imgd))[1] / 1
        # z-score with exponent
        imgdz = (imgd / z0) ** w

        # get gradient

        sx = cv.Sobel(imgd, cv.CV_64F, 1, 0, ksize=5)
        sy = cv.Sobel(imgd, cv.CV_64F, 0, 1, ksize=5)

        sx[np.isclose(sx, 0)] = 0.5
        sy[np.isclose(sx, 0)] = 0.5

        v2x = np.nan_to_num(sx / np.sqrt(sy ** 2 + sx ** 2))
        v2y = np.nan_to_num(sy / np.sqrt(sy ** 2 + sx ** 2))

        v1x = -v2y
        v1y = v2x

        # normalize zscores range btw 0.5 and 1
        lam1 = imgdz / (imgdz + 1)

        lam1[np.isclose(lam1, 0)] = 0.5
        lam2 = 1. - lam1

        lam3 = lam2*mu

        lams = lam1 + lam2 + lam3

        lam1 = lam1 / lams
        lam2 = lam2 / lams
        lam3 = lam3 / lams

        for i in range(dim):
            for j in range(dim):

                v1ij = np.array([v1x[i, j], v1y[i, j], 0])
                v2ij = np.array([v2x[i, j], v2y[i, j], 0])
                v3ij = np.array([0, 0, 1])

                dt[:, :, i, j, k] = lam1[i, j]*np.outer(v1ij, v1ij) +\
                                  lam2[i, j]*np.outer(v2ij, v2ij) +\
                                  lam3[i, j]*np.outer(v3ij, v3ij)

    return dt


def img_to_2dti(img):

    # set minimum to zero
    img = img - np.min(img)
    # get the least
    z0 = np.sort(np.unique(img))[1] / 2

    # get gradient

    sx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    v2x = np.nan_to_num(sx / np.sqrt(sy ** 2 + sx ** 2))
    v2y = np.nan_to_num(sy / np.sqrt(sy ** 2 + sx ** 2))

    v1x = -v2y
    v1y = v2x

    lam1 = (img / z0) / (img / z0 + 1)
    lam1[lam1 == 0] = 0.5
    lam2 = 1. - lam1

    n = img.shape[0]

    dt = np.zeros([2, 2, n, n])

    for i in range(n):
        for j in range(n):

            v1ij = np.array([v1x[i, j], v1y[i, j]])
            v2ij = np.array([v2x[i, j], v2y[i, j]])

            dt[:, :, i, j] = lam1[i, j]*np.outer(v1ij, v1ij) + lam2[i, j]*np.outer(v2ij, v2ij)

    return dt


def save_tensor(dt):

    base = '/Users/cassiano/Dropbox/cob/work/upenn/research/projects/defri/dnn/code/dti-mnist-proto/readMNIST/' +\
           'plotDTI/plotDTI'
    fname = 'dt.mat'
    sio.savemat(base + '/' + fname, {'dt': dt})


def plot_quiv(sx, sy):
    plt.quiver(sx, sy)


def main():

    batch_size = 64
    lift_dim = 6

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):

        dts = []

        for k in range(batch_size):

            img = data[k, 0, :, :].numpy()

            print(target[k])

            dts.append(img_to_3dti(img, lift_dim))

        save_tensor(dts)


if __name__ == '__main__':
    main()
