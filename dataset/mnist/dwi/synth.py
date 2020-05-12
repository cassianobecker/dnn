import os
import numpy.linalg as la
import numpy as np

from os.path import join, expanduser
from dipy.io import read_bvals_bvecs
from dipy.io.image import save_nifti

from scipy.spatial.transform import Rotation

rel_path = '~/.dnn/datasets/synth'
axes = [[1, 0, 0], [0, 1, 1]]
name = 'synth'
width, height, depth = 30, 32, 34
num_directions = 16


class Ellipse:

    def __init__(self, img_dims):
        sigma = 5
        cov0 = np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.01]])
        R = Rotation.from_rotvec([np.pi / 3, np.pi / 4, np.pi / 5]).as_matrix()
        cov = (sigma) * (R @ cov0 @ R.T)
        center = (np.array(img_dims) / 2).astype(np.int)

        self.cov = cov
        self.norm_cov = np.linalg.norm(cov)
        self.inv_cov = np.linalg.pinv(self.cov)
        self.mu = center

    def _quad_form_at(self, voxel):
        delta = np.array(voxel) - self.mu
        quad_form = delta.T @ self.inv_cov @ delta
        return quad_form

    def exp_quad(self, voxel):
        return np.exp(-self._quad_form_at(voxel))

    def quad_exp_quad(self, voxel, direction):
        mult = self._quad_form_at(direction) / self.norm_cov
        return mult * self.exp_quad(voxel)


def generate_image():

    print('generating image... ', end='')

    img_dims = (width, height, depth)
    img, affine = image(img_dims)

    save_image(img, affine, name)

    print('done.')


def generate_ground_truth_movement():

    print('generating ground truth movement... ', end='')

    img_dims = (width, height, depth)

    bvals, bvecs = load_bvals_bvecs_from_hcp_template()
    bvals = bvals[:num_directions]
    bvecs = bvecs[:num_directions, :]
    save_bvals_bvecs(bvals, bvecs)

    mov, affine = movement_for_all_directions(img_dims, bvecs)
    save_mov(mov, affine, name)

    print(f'done.')


def load_bvals_bvecs_from_hcp_template():

    template_subject = '100206'
    relative_path = '~/.dnn/datasets/hcp/mirror/HCP_1200'
    subject_path = join(expanduser(relative_path), template_subject, 'T1w', 'Diffusion')
    bvals_url = join(subject_path, 'bvals')
    bvecs_url = join(subject_path, 'bvecs')
    bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)

    return bvals, bvecs


def load_bvals_bvecs():

    path = os.path.expanduser(rel_path)

    bvals_url = os.path.join(path, 'bvals')
    bvecs_url = os.path.join(path, 'bvecs')
    bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)

    return bvals, bvecs


def save_image(img, affine, name):

    path = os.path.expanduser(rel_path)
    url = os.path.join(path, name)

    if not os.path.isdir(path):
        os.makedirs(path)

    save_nifti(url + '.nii.gz', img, affine)


def save_mov(mov, affine, name):

    path = os.path.expanduser(rel_path)
    url = os.path.join(path, name)

    if not os.path.isdir(path):
        os.makedirs(path)
    save_nifti(url + '_mov_truth.nii.gz', mov, affine)


def save_bvals_bvecs(bvals, bvecs):

    path = os.path.expanduser(rel_path)

    fmt_bvals = '%d'
    delimiter = ' '
    url_bvals = os.path.join(path, 'bvals')
    np.savetxt(url_bvals, np.expand_dims(bvals, axis=0), fmt=fmt_bvals, delimiter=delimiter)

    fmt_bvecs = '%2.6f'
    delimiter = '  '
    url_bvecs = os.path.join(path, 'bvecs')
    np.savetxt(url_bvecs, bvecs.T, fmt=fmt_bvecs, delimiter=delimiter)


def mov_img(dims, direction):

    ellipse = Ellipse(dims)

    mov = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                voxel = (i, j, k)
                mov_ijk = ellipse.quad_exp_quad(voxel, direction)
                mov[i, j, k] = mov_ijk

    return mov


def image(dims):

    img = np.zeros(dims)

    ellipse = Ellipse(dims)

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                voxel = (i, j, k)
                img_ijk = ellipse.exp_quad(voxel)
                img[i, j, k] = img_ijk

    affine = np.eye(4)

    return img, affine


def movement_for_all_directions(img_dims, bvecs):

    bvec_list = bvecs.tolist()
    movs = []
    for k, direction in enumerate(bvec_list):
        print(f'direction {k + 1} of {len(bvec_list)}')
        mov_for_direction = mov_img(img_dims, direction)
        movs.append(mov_for_direction)

    mov = np.transpose(np.array(movs), (1, 2, 3, 0))

    affine = np.eye(4)

    return mov, affine


if __name__ == '__main__':
    generate_image()
    generate_ground_truth_movement()
