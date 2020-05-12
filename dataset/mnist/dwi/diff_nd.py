import numpy as np
from skimage.draw import line_nd
import matplotlib.pyplot as plt


def pad_image(image, radius):
    return np.pad(image, pad_width=radius, mode='constant', constant_values=0)


def successive_differences(image, direction, radius, function_over_diff=None, normalize=True):
    # normalize direction
    direction = np.array(direction) / np.linalg.norm(np.array(direction))

    # pad image
    padded_image = pad_image(image, radius)

    # compute initial point
    start = [radius] * 3

    # compute final point
    stop = np.add(start, np.array(direction * radius)).astype(np.int).tolist()

    # get set of coordinates connecting start point to stop point
    line_coords = line_nd(start, stop)
    points = np.array(line_coords).T.tolist()

    diffs = np.zeros_like(image)

    if function_over_diff is None:
        # identity function
        function_over_diff = lambda x: x

    k = 0
    for first, second in zip(points[:-1], points[1:]):

        first_image = padded_image[slice_for(first, image.shape)]
        second_image = padded_image[slice_for(second, image.shape)]
        # diff = second_image - first_image
        diff = second_image
        # if normalize is True:
        #     diff = np.divide(diff, first_image, out=np.zeros_like(diff), where=first_image != 0)

        diffs += function_over_diff(diff) * np.exp(-0.5*k)
        k = k + 1

    return diffs


def slice_for(point, dims):
    return tuple(slice(point[k], point[k] + dims[k]) for k in range(3))


def test_diff_nd():
    axis = [0.3, 0.4, 0.6]
    axis = np.array(axis) / np.linalg.norm(np.array(axis))

    width, height, depth = 10, 15, 20
    grid = np.mgrid[:width, :height, :depth]

    image = np.einsum('i...,i->...', grid, axis)

    direction = axis
    radius = 10

    image_weight = successive_differences(image, direction, radius, np.abs)

    pass


if __name__ == '__main__':
    test_diff_nd()
