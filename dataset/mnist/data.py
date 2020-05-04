from __future__ import print_function
# from future.standard_library import install_aliases

# install_aliases()

import os
import gzip
import struct
import array
import numpy as np
from urllib.request import urlretrieve
from util.path import absolute_path

relative_path = '~/.dnn/datasets/mnist'
mnist_path = absolute_path(os.path.join(os.path.expanduser(relative_path), 'data'))


def download(url, filename):

    if not os.path.exists(mnist_path):
        os.makedirs(mnist_path)
    out_file = os.path.join(mnist_path, filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images(mnist_path + '/train-images-idx3-ubyte.gz')
    train_labels = parse_labels(mnist_path + '/train-labels-idx1-ubyte.gz')
    test_images = parse_images(mnist_path + '/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels(mnist_path + '/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist(onehot=True):
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0
    if onehot:
        train_labels = one_hot(train_labels, 10)
        test_labels = one_hot(test_labels, 10)

    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels
