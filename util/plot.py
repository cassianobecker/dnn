import matplotlib.pyplot as plt
import numpy as np


def plot_mid_slices(tensor):
    slices = (np.array(tensor.shape)/2).astype(np.int)
    plot_tensor_slices(tensor, slices)


def plot_tensor_slices(tensor, slices=None, middle=False):

    # 0, 1 coordinates: left-right; x, fix x -> sagittal view
    # 2, 3 coordinates: front-back; z, fix z -> coronal view
    # 4, 5 coordinates: top-bottom; y, fix y -> transverse view

    if slices is None and middle is True:
        slices = [int(tensor.shape[0]/2), int(tensor.shape[1]/2), int(tensor.shape[2]/2)]

    n_rows = 2
    n_cols = 2

    plt.figure()
    plt.subplot(n_rows, n_cols, 2)
    img = np.squeeze(tensor[slices[0], :, :]).T
    img = np.flip(img, 0)
    img = np.flip(img, 1)
    plt.imshow(img)
    plt.title('sagittal')
    plt.colorbar()
    plt.autoscale(enable=True, axis='both', tight=None)

    plt.subplot(n_rows, n_cols, 3)
    img = np.squeeze(tensor[:, slices[1], :]).T
    img = np.flip(img, 0)
    plt.imshow(img)
    plt.title('coronal')
    plt.colorbar()
    plt.autoscale(enable=True, axis='both', tight=None)

    plt.subplot(n_rows, n_cols, 4)
    img = np.squeeze(tensor[:, :, slices[2]])
    img = np.flip(img, 1)
    plt.title('transversal')
    plt.colorbar()
    plt.autoscale(enable=True, axis='both', tight=None)

    plt.imshow(img)
