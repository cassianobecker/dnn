import numpy as np
from os.path import join, expanduser, isdir
import os
from shutil import copyfile

import plotly.graph_objects as go

from dataset.mnist.data import load_mnist
from dataset.mnist.dwi.tube import tubify, get_img
from dataset.mnist.dwi.gradient import grad_filter_3d, dir_grad_3d, bvecs_grad

from dipy.io.image import save_nifti
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs


def make_3d():

    max_digits = 10
    img_depth = 15

    n, images, labels, test_images, test_labels = load_mnist()

    template_subject = '100206'
    relative_path = '~/.dnn/datasets/hcp/mirror/HCP_1200'
    subject_path = join(expanduser(relative_path), template_subject, 'T1w', 'Diffusion')
    bvals_url = join(subject_path, 'bvals')
    bvecs_url = join(subject_path, 'bvecs')
    bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)

    print(f'reading data from template subject {template_subject} ... ', end='')

    # data, affine, img = load_nifti(join(expanduser(subject_path), 'data.nii.gz'), return_img=True)

    affine = np.eye(4)

    print('done.')

    for digit_idx in range(max_digits):

        print(f'processing digit {digit_idx} ... ', end='')

        img = get_img(images, digit_idx)
        img3d = tubify(img, img_depth)
        grads = grad_filter_3d(img3d)
        bvecs_grads = bvecs_grad(grads, bvecs)

        bvecs_grads = np.einsum('ijkl,kij->ijkl', bvecs_grads, img3d)

        # bvecs_grads = np.transpose(np.divide(np.transpose(img3d, (2, 1, 0)), np.transpose(bvecs_grads, (3, 0, 1, 2)) +
        #                                      0.01), (1, 2, 3, 0))

        processing_relative_path = '~/.dnn/datasets/hcp/processing_mnist/mnist/'
        processing_path = join(expanduser(processing_relative_path), f'{digit_idx}')
        if not isdir(processing_path):
            os.makedirs(processing_path)

        save_nifti(join(processing_path, 'data.nii.gz'), bvecs_grads, affine)
        copyfile(bvecs_url, join(processing_path, 'bvecs'))
        copyfile(bvals_url, join(processing_path, 'bvals'))

        print('done.')

    pass


def test_one_direction():

    digit_idx = 1
    img_depth = 15

    n, images, labels, test_images, test_labels = load_mnist()

    img = get_img(images, digit_idx)
    img3d = tubify(img, img_depth)
    plot_volume(img3d)

    direction = [0.2, 0.4, -0.8]
    grads = grad_filter_3d(img3d)
    dir_grad = dir_grad_3d(grads, direction)

    plot_volume(dir_grad)


def plot_volume(values):

    x, y, z = np.mgrid[0:values.shape[0], 0:values.shape[1], 0:values.shape[2]]

    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        # isomin=-0.1,
        # isomax=0.8,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=21,  # needs to be a large number for good volume rendering
        # colorscale='RdBu'
        ))
    fig.show()


if __name__ == '__main__':
    # test_one_direction()
    make_3d()
