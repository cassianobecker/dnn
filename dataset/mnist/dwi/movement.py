import os
import numpy.linalg as la
import numpy as np
from skimage.draw import line_nd

from os.path import join, expanduser
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

rel_path = '~/.dnn/datasets/synth'
name = 'synth'


def process_movement():

    bvals, bvecs = load_bvals_bvecs()
    img, affine = load_image_from_nifti()
    mov = get_movement_estimates(img, bvecs)
    save_mov_image(mov, affine, name)


def load_image_from_numpy():
    path = os.path.expanduser(rel_path)
    url = os.path.join(path, name + '.npz')
    img_dict = np.load(url, allow_pickle=True)
    return img_dict['img']


def load_image_from_nifti():
    base_path = expanduser(rel_path)
    digit_hardi_url = join(base_path,  name + '.nii.gz')
    img, affine = load_nifti(digit_hardi_url)
    return img, affine


def load_bvals_bvecs():
    path = os.path.expanduser(rel_path)
    bvals_url = join(path, 'bvals')
    bvecs_url = join(path, 'bvecs')
    bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)

    return bvals, bvecs


def save_mov_image(mov, affine, name):

    path = os.path.expanduser(rel_path)

    if not os.path.isdir(path):
        os.makedirs(path)

    # np.savez(os.path.join(path, name + '_mov'), mov=mov)
    save_nifti(os.path.join(path, name + '_mov.nii.gz'), mov, affine)


def mov_img(img, direction):

    mov = np.zeros_like(img)
    dims = img.shape

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                mov_ijk = movement(img, (i, j, k), direction, radius=10, eps=0.01)
                mov[i, j, k] = mov_ijk

    return mov


def movement(img, center, direction, radius=10, eps=0.01, min_val=1e-9):

    center_value = img[center[0], center[1], center[2]]

    mov = 0

    if abs(center_value) > min_val:

        coords = get_points_bidirectional(center, direction, radius, img.shape)
        z = img[coords[0], coords[1], coords[2]]

        if len(z) > 1:
            deltas = np.abs(z[0] - z[1:]) + eps
            variation = (1 / (len(z) - 1)) * np.sum(deltas)

            mov = center_value / variation

    return mov


def get_movement_estimates(img, bvecs, max_bvecs=None):

    bvec_list = bvecs.tolist()[:max_bvecs]

    movs = []
    for k, direction in enumerate(bvec_list):
        print(f'direction {k +1} of {len(bvec_list)}')
        mov_for_direction = mov_img(img, direction)
        movs.append(mov_for_direction)

    mov = np.transpose(np.array(movs), (1, 2, 3, 0))

    return mov





if __name__ == '__main__':
    process_movement()
