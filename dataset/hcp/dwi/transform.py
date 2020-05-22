import dipy
from dipy.io.image import load_nifti, save_nifti
import os
from dipy.align.imaffine import AffineMap
import numpy as np
import numpy.random as npr
from scipy.spatial.transform import Rotation as R


def rotate_tensor(x, angles, shift=None):

    rot = R.from_euler('zyx', angles).as_dcm()
    x = rotate_dti_outside(x, rot, shift=shift)
    x = rotate_dti_inside(x, rot)

    return x


def rotate_dti_outside(x, rot, shift=None):

    shift = np.zeros((3, 1)) if shift is None else np.expand_dims(shift, axis=1)

    affine_rot = np.block([[rot, shift], [np.zeros((1, 3)), 1]])

    affine_transform = AffineMap(affine_rot, domain_grid_shape=x.shape[:-1])

    return np.array([affine_transform.transform(x[..., k]) for k in range(6)]).transpose((1, 2, 3, 0))


def to_mat(x):

    col1 = np.stack((x[..., 0], x[..., 1], x[..., 2]), axis=0).transpose((1, 2, 3, 0))
    col2 = np.stack((x[..., 1], x[..., 3], x[..., 4]), axis=0).transpose((1, 2, 3, 0))
    col3 = np.stack((x[..., 2], x[..., 4], x[..., 5]), axis=0).transpose((1, 2, 3, 0))

    return np.stack((col1, col2, col3), axis=0).transpose((1, 2, 3, 4, 0))


def to_vec(x):

    # Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
    return np.stack(
        (x[..., 0, 0], x[..., 0, 1], x[..., 0, 2], x[..., 1, 1], x[..., 1, 2], x[..., 2, 2])
    ).transpose((1, 2, 3, 0))


def rotate_dti_inside(x, rot):
    return to_vec(np.einsum('lx,ijkxy,ym->ijklm', rot, to_mat(x), rot.T))
