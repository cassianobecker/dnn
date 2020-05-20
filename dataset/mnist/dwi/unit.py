import os

import dipy.reconst.dti as dti
import numpy as np
from dipy.core.gradients import gradient_table, generate_bvecs
from dipy.io.image import save_nifti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import auto_response
from scipy.spatial.transform import Rotation


class Ellipse:

    def __init__(self, mean, covariance):
        self.covariance = covariance
        self.inv_covariance = np.linalg.pinv(self.covariance)
        self.mean = mean

    def quad_form_at(self, x):
        delta = x - self.mean
        quad_form = delta.T @ self.covariance @ delta
        return quad_form

    def exp_quad_at(self, x):
        return np.exp(-self.quad_form_at(x))

    def rotate_covariance(self, angles):
        R = Rotation.from_rotvec(angles).as_matrix()
        self.covariance = R @ self.covariance @ R.T
        self.inv_covariance = R @ self.inv_covariance @ R.T


class DiffusionImageModel:

    def __init__(self, img_dims, intensity_threshold=1.e-3, rotate=False):

        center = np.array(img_dims) / 2
        cov = (1 / 5) * np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.01]])
        ellipse = Ellipse(mean=center, covariance=cov)

        if rotate is True:
            angles = [np.pi / 3, np.pi / 4, np.pi / 5]
            ellipse.rotate_covariance(angles)

        self.ellipse = ellipse
        self.img_dims = img_dims
        self.intensity_threshold = intensity_threshold

        self.affine = np.eye(4)
        self.image = None

        self.b0_image = None
        self.bvals = None
        self.bvecs = None

        self.volumes = None

        self.dti = None
        self.v1 = None
        self.odf = None

    def fit_dti(self):

        covariance = [
            [1., 0., 0.],
            [0., 0.2, 0.],
            [0., 0., 0.2]
        ]
        covariance = 100 * np.array(covariance)

        dti_tensor = np.zeros((1, 1, 1, 3, 3))
        dti_tensor[0, 0, 0, :, :] = covariance
        diff_coeffs = dti.lower_triangular(dti_tensor)

        # reorder coefficients to FDT convention Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
        # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide
        self.dti = diff_coeffs[:, :, :, (0, 1, 3, 2, 4, 5)]

        v1_tensor = np.zeros((1, 1, 1, 3))
        v1 = np.linalg.eigh(covariance)[1][:, -1]
        v1_tensor[0, 0, 0, :] = v1
        self.v1 = v1_tensor

    def save_dti(self, relative_path):

        name = 'dti.nii.gz'

        path = os.path.expanduser(relative_path)
        url = os.path.join(path, name)

        if not os.path.isdir(path):
            os.makedirs(path)

        save_nifti(url, self.dti, self.affine)

        name = 'v1.nii.gz'

        path = os.path.expanduser(relative_path)
        url = os.path.join(path, name)

        save_nifti(url, self.v1, self.affine)


def _canonical_bvecs():
    sq2 = np.sqrt(2) / 2
    bvecs = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [sq2, -sq2, 0],
                      [sq2, 0, -sq2],
                      [0, sq2, sq2]],
                     [0, sq2, -sq2])
    return bvecs


def process_diffusion_image():

    relative_path = '~/.dnn/datasets/unit'

    width, height, depth = 1, 1, 1

    image = DiffusionImageModel([width, height, depth], rotate=True)

    image.fit_dti()
    image.save_dti(relative_path)


if __name__ == '__main__':
    process_diffusion_image()
