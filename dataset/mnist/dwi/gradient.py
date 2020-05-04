import numpy as np
import scipy.ndimage


def sobel_3d():

    hx = np.zeros((3, 3, 3))
    hy = np.zeros((3, 3, 3))
    hz = np.zeros((3, 3, 3))

    hx[:, :, 0] = [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]
    hx[:, :, 1] = [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]]
    hx[:, :, 2] = [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]

    hy[:, :, 0] = [[-1, -3, -1], [0, 0, 0], [1, 3, 1]]
    hy[:, :, 1] = [[-3, -6, -3], [0, 0, 0], [3, 6, 3]]
    hy[:, :, 2] = [[-1, -3, -1], [0, 0, 0], [1, 3, 1]]

    hz[:, :, 0] = [[-1, -3, -1], [-3, -6, -3], [-1, -3, -1]]
    hz[:, :, 1] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    hz[:, :, 2] = [[1, 3, 1], [3, 6, 3], [1, 3, 1]]

    return hx, hy, hz


def grad_filter_3d(img):

    mode = 'reflect'

    hx, hy, hz = sobel_3d()

    gx = scipy.ndimage.correlate(img, hx, mode=mode).transpose()
    gy = scipy.ndimage.correlate(img, hy, mode=mode).transpose()
    gz = scipy.ndimage.correlate(img, hz, mode=mode).transpose()

    eps = 15.

    gx = 1 / (gx + eps)
    gy = 1 / (gy + eps)
    gz = 1 / (gz + eps)

    grads = np.array([gx, gy, gz])

    return grads


def dir_grad_3d(grads, direction):

    # normalize direction vector
    direction = direction / np.linalg.norm(direction)
    # apply inner product
    return np.einsum('lijk,l->ijk', grads, direction)


def bvecs_grad(grads, bvecs):

    bvecs_grads = []
    for direction in bvecs.tolist():
        bvecs_grads.append(dir_grad_3d(grads, direction))

    return np.transpose(np.array(bvecs_grads), (1, 2, 3, 0))
