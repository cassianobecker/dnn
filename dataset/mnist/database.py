import numpy as np
from dataset.mnist.data import load_mnist
import os
from urllib.request import urlretrieve


class MnistDatabase:

    def __init__(self):

        _, train_images, train_labels, test_images, test_labels = load_mnist()

        self.images = dict()
        self.labels = dict()

        self.images['train'] = train_images
        self.images['test'] = test_images

        self.labels['train'] = train_labels
        self.labels['test'] = test_labels

    def get_image(self, image_idx, regime='train'):

        image = self.images[regime][image_idx, :]
        label = self.labels[regime][image_idx]

        return np.reshape(image, (28, 28)), label

    def image_batch(self, batch_index, number_of_batches, regime):
        batch_size = int(self.number_of_images(regime) / number_of_batches)
        initial = int(batch_index * batch_size)
        final = int((batch_index + 1) * batch_size)
        return range(self.number_of_images(regime))[initial:final]

    def number_of_images(self, regime):
        return self.images[regime].shape[0]


class KmnistDatabase:

    def __init__(self):
        relative_path = '~/.dnn/datasets/kmnist/mirror'
        self.path = os.path.expanduser(relative_path)

        if self.check_downloaded() is not True:
            self.download()

        self.images = dict()
        self.labels = dict()

        self.images['train'] = np.load(os.path.join(self.path, 'kmnist-train-imgs.npz'))['arr_0']
        self.images['test'] = np.load(os.path.join(self.path, 'kmnist-test-imgs.npz'))['arr_0']

        self.labels['train'] = np.load(os.path.join(self.path, 'kmnist-train-labels.npz'))['arr_0']
        self.labels['test'] = np.load(os.path.join(self.path, 'kmnist-test-labels.npz'))['arr_0']

    def get_image(self, image_idx, regime='train'):

        image = self.images[regime][image_idx, :, :] / 255
        label = np.zeros(10)
        label[self.labels[regime][image_idx]] = 1
        return image, label

    def number_of_images(self, regime):
        return self.images[regime].shape[0]

    def check_downloaded(self):
        return os.path.isdir(self.path)

        return downloaded

    def download(self):

        base_url = 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/'

        files = ['kmnist-train-imgs.npz',
                 'kmnist-train-labels.npz',
                 'kmnist-test-imgs.npz',
                 'kmnist-test-labels.npz']

        print('Downloading Kmnist files... ')

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for k, file in enumerate(files):

            print(f'file ({k + 1} of {len(files)}): {file} ')

            local = os.path.join(self.path, file)
            remote = os.path.join(base_url, file)

            if not os.path.isfile(local):
                urlretrieve(remote, local)

        print('done.')

    def image_batch(self, batch_index, number_of_batches, regime):

        batch_size = int(self.number_of_images(regime) / number_of_batches)
        initial = int(batch_index * batch_size)
        final = int((batch_index + 1) * batch_size)
        return range(self.number_of_images(regime))[initial:final]
