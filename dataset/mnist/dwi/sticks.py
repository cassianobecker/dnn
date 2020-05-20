import os
import numpy as np
from scipy import signal

from dipy.io.image import save_nifti
from skimage.draw import line_nd


def create_stick_figure(width, height, depth):

    img = np.zeros((width, height, depth), dtype=np.double)

    line_coords = line_nd((0, height/2, 0), (width-1, height/2, depth-1))
    img[line_coords[0], line_coords[1], line_coords[2]] = 1

    # line_coords2 = line_nd((0, height/2, depth/2), (width-1, height/2, depth/2))
    # img[line_coords2[0], line_coords2[1], line_coords2[2]] = 1

    img = img / np.max(img)

    return img


def convolve_tube(img):

    sigma = 1.5    # width of kernel

    x = np.arange(-3, 4, 1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3, 4, 1)
    z = np.arange(-3, 4, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))

    img = signal.convolve(img, kernel, mode="same")

    return img


def save_image(img, affine, name):

    rel_path = '~/.dnn/datasets/sticks'

    path = os.path.expanduser(rel_path)

    if not os.path.isdir(path):
        os.makedirs(path)

    url = os.path.join(path, name)

    np.savez(url, img=img)

    save_nifti(url + '.nii.gz', img, affine)


def generate_figures():

    name = 'sticks1'

    width = 30
    height = 30
    depth = 30

    img = create_stick_figure(width, height, depth)
    img = convolve_tube(img)

    affine = np.eye(4)
    save_image(img, affine, name)


if __name__ == '__main__':

    generate_figures()
