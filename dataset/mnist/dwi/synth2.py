import os

import dipy.reconst.dti as dti
import numpy as np
from dipy.core.gradients import gradient_table, generate_bvecs
from dipy.io.image import save_nifti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import auto_response
from scipy.spatial.transform import Rotation
from skimage.draw import line_nd
from dataset.mnist.processor.diff_nd import successive_differences
import matplotlib.pyplot as plt
from dataset.mnist.data import load_mnist
from dataset.mnist.processor.tube import tubify, get_img

bval0 = 100
radius = 10


def process_diffusion_image():


    relative_path = '~/.dnn/datasets/ellipse'
    # width, height, depth = 30, 32, 34
    # image = DiffusionImageModel([width, height, depth], rotate=True)
    # image.generate_image()

    depth = 8

    # relative_path = '~/.dnn/datasets/mnist'
    # image_idx_mnist = 7
    # N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    # image = get_img(train_images, image_idx_mnist)

    relative_path = '~/.dnn/datasets/kmnist'
    kmnist_url = '/Users/cassiano/.dnn/datasets/kmnist/kmnist.npz'
    data = np.load(kmnist_url)
    image_idx_kmnist = 32 #49 # 32, 77
    image = data['arr_0'][image_idx_kmnist, :, :] / 255


    import time
    # for k in range(200):
    #     image_idx_kmnist = k
    #     image = data['arr_0'][image_idx_kmnist, :, :] / 255
    #     plt.imshow(image)
    #     plt.title(f'{k}')
    #     plt.show()
    #
    #     # time.sleep(2)
    #     plt.cla()

    image3d = tubify(image, depth)

    width, height, depth = image3d.shape
    image = DiffusionImageModel([width, height, depth], rotate=True)
    image.image = image3d
    image.make_mask()

    image.save_image(relative_path)

    image.set_b0_image(same=True, weight=3000)
    image.generate_bvals_bvecs(n_bvals=200, bval_mult=bval0)
    # image.generate_bvals_bvecs(canonical=False, bval_mult=1)
    image.save_bvals_bvecs(relative_path)

    # region_size = [region_radius] * 3
    # corner = (np.array(image.dims) / 2 - np.array(region_size) / 2).astype(np.int).tolist()

    # region_size = image.dims
    # corner = [0, 0, 0]
    # image.set_diffusion_region(corner, region_size)


    image.generate_diffusion_volumes()
    image.save_diffusion_volumes(relative_path)

    image.fit_dti()
    # print(image.dti[image.region])
    image.save_dti(relative_path)

    image.fit_odf()
    image.save_odf(relative_path)
    pass


class DiffusionImageModel:

    def __init__(self, img_dims, intensity_threshold=1.e-3, rotate=False):

        center = np.array(img_dims) / 2
        cov = np.linalg.pinv(np.array([[10, 0, 0], [0, 5, 0], [0, 0, 1]]))

        ellipse = Ellipse(mean=center, covariance=cov)

        if rotate is True:
            angles = [np.pi / 3, np.pi / 4, np.pi / 5]
            ellipse.rotate_covariance(angles)

        self.ellipse = ellipse

        self.dims = img_dims
        self.image = None

        self.corner = None
        self.region_size = None
        self.region = None

        self.affine = np.eye(4)
        self.mask = None

        self.intensity_threshold = intensity_threshold

        self.b0_image = None
        self.bvals = None
        self.bvecs = None

        self.volumes = None

        self.dti = None
        self.v1 = None
        self.evals = None
        self.odf = None

    # def set_diffusion_region(self, corner, region_size):
    #     self.corner = corner
    #     self.region_size = region_size
    #     self.region = tuple(slice(self.corner[k], self.corner[k] + self.region_size[k]) for k in range(3))
    #
    #     diffusion_mask = np.zeros_like(self.image)
    #     diffusion_mask[self.region] = 1
    #     self.mask = self.mask * diffusion_mask
    #
    # def _quad_form_empirical(self, voxel_coords, direction, radius=10):
    #
    #     def get_points_along_direction(center, _direction):
    #         # _direction = _direction / np.linalg.norm(_direction)
    #         endpoint = np.array(center) + (radius * _direction).astype(np.int)
    #         endpoint = np.minimum(endpoint, np.array(self.dims) - 1).astype(np.int)
    #         endpoint = np.maximum(endpoint, np.zeros(3)).astype(np.int)
    #         _coords = line_nd(center, endpoint.tolist())
    #         return _coords
    #
    #     def successive_differences(z):
    #         z = [z for z in z.tolist() if z > self.intensity_threshold]
    #         z = np.array(z)
    #         _deltas = (z[0] - z[1:]) if len(z) > 1 else [0]
    #         return _deltas
    #
    #     def differences_for_direction(_direction):
    #         coords = get_points_along_direction(voxel_coords, _direction)
    #         _z = self.image[coords[0], coords[1], coords[2]]
    #         return successive_differences(_z)
    #
    #     deltas = [*differences_for_direction(direction), *differences_for_direction(-direction)]
    #
    #     # experimentally, found btw lam_min = 1e-4 and lam_max = 5e-3
    #
    #     lam_min = 1.e-4
    #     lam_max = 5.e-3
    #     max_var = 1000
    #
    #     relative_mean_variation = np.linalg.norm(deltas) / (len(deltas) * max_var)
    #
    #     if relative_mean_variation > 0.95:
    #         print('##### close to limit ratio **************\n\n\n\n\n\n')
    #     quad_form_value = lam_max - (lam_max - lam_min) * relative_mean_variation
    #
    #     corrected_quad_form_value = quad_form_value
    #
    #     return corrected_quad_form_value
    #
    # def _quad_form_synth(self, voxel_coords, direction):
    #     D_ijk = self.ellipse.covariance * (self.image[voxel_coords])
    #     quad_form_at_bvec = direction.T @ D_ijk @ direction
    #     return quad_form_at_bvec
    #
    # def _diffusion_weight_at_voxel_for_direction(self, voxel_coords, bval, bvec):
    #
    #     # quad_form_function = self._diffusion_exponent_synth
    #     quad_form_function = self._quad_form_empirical
    #
    #     if self.b0_image is None:
    #         raise ValueError('Please set an image at b0 value using method "set_b0_image"')
    #
    #     dw = 0
    #     if self.image[voxel_coords] > 0:
    #         quad_form_value = quad_form_function(voxel_coords, np.array(bvec))
    #         dw = self.b0_image[voxel_coords] * np.exp(-bval * quad_form_value)
    #
    #     return dw

    # def _diffusion_volume_at_direction(self, bval, bvec):
    #
    #     def _range(_slice):
    #         return range(_slice.start, _slice.stop)
    #
    #     vol = np.zeros(self.dims)
    #     for i in _range(self.region[0]):
    #         for j in _range(self.region[1]):
    #             for k in _range(self.region[2]):
    #                 voxel_coords = (i, j, k)
    #                 vol_ijk = self._diffusion_weight_at_voxel_for_direction(voxel_coords, bval, bvec)
    #                 vol[i, j, k] = vol_ijk
    #
    #     # print(vol[self.region])
    #     return vol

    # @staticmethod
    # def adc_for_diffs(diffs, max_variation):
    #
    #     lam_min = 1.e-4
    #     lam_max = 5.e-3
    #     enhancement = 1
    #
    #     relative_mean_variation = diffs * enhancement / max_variation
    #     adc = lam_max - (lam_max - lam_min) * relative_mean_variation
    #
    #     return adc

    def _diffs_at_direction_tensor(self, bval, bvec, radius=3):

        diffs = successive_differences(self.image, bvec, radius, normalize=True) + \
                successive_differences(self.image, -bvec, radius, normalize=True)

        return diffs

    # def maximum_variation_along_axis(self):
    #
    #     directions = np.eye(3).tolist()
    #     radius_for_max = 2
    #
    #     diffs_for_axis = [np.max(successive_differences(self.image, direction, radius_for_max, np.abs, normalize=True))
    #                       for direction in directions]
    #
    #     max_diff = np.max(np.array(diffs_for_axis))
    #
    #     return max_diff

    def adc_fun(self, x):
        return 1 / (1 + np.exp(-x))

    def generate_diffusion_volumes(self):

        diffusion_volumes = []

        for i, bvec in enumerate(self.bvecs.tolist()):

            print(f'generating volume {i + 1} of {len(self.bvals)} ... ', end='')

            bval = self.bvals[i]
            volume = self._diffs_at_direction_tensor(bval, np.array(bvec), radius)

            lam_min = 2.e-4
            lam_max = 5.e-3
            adc = lam_min + volume * (lam_max - lam_min) / 1

            diffusion_volume = self.b0_image * np.exp(-bval * adc)

            diffusion_volumes.append(diffusion_volume)

            print('done.')

        self.volumes = np.array(np.transpose(diffusion_volumes, (1, 2, 3, 0)))

    def set_b0_image(self, same=False, weight=1):

        if same is True:
            b0_image = self.image
        else:
            b0_image = np.ones_like(self.image)

        self.b0_image = weight * b0_image

    def intensity_at_voxel(self, voxel_coords):
        intensity = self.ellipse.exp_quad_at(np.array(voxel_coords), sigma=10)
        return intensity if intensity > self.intensity_threshold else 0

    def generate_image(self, mask_threshold=3.e-2):

        img = np.zeros(self.dims)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    voxel_coords = (i, j, k)
                    img_ijk = self.intensity_at_voxel(voxel_coords)
                    img[i, j, k] = img_ijk

        self.image = img

    def reweight_image(self, weight):
        self.image = weight * self.image

    def make_mask(self, threshold=1e-3):
        mask = np.zeros_like(self.image)
        mask[self.image > threshold] = 1
        self.mask = mask.astype(np.int)

    def generate_bvals_bvecs(self, canonical=False, n_bvals=64, bval_mult=1, n_b0=8):

        if canonical is True:
            n_bvals = 7
            print(f'bvals: using {n_bvals} canonical directions.')
            bvals = bval_mult * np.ones(n_bvals)
            bvecs = self._canonical_bvecs()
        else:
            bvals = bval_mult * np.ones(n_bvals)
            bvecs = generate_bvecs(n_bvals, 1)

        if self.b0_image is not None:
            bvecs = np.concatenate((np.zeros((n_b0, 3)), bvecs), axis=0)
            bvals = np.concatenate((np.zeros(n_b0), bvals))

        self.bvals = bvals
        self.bvecs = bvecs

    @staticmethod
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

    def save_bvals_bvecs(self, rel_path):

        path = os.path.expanduser(rel_path)

        fmt_bvals = '%d'
        delimiter = ' '
        url_bvals = os.path.join(path, 'bvals')
        np.savetxt(url_bvals, np.expand_dims(self.bvals, axis=0), fmt=fmt_bvals, delimiter=delimiter)

        fmt_bvecs = '%2.6f'
        delimiter = '  '
        url_bvecs = os.path.join(path, 'bvecs')
        np.savetxt(url_bvecs, self.bvecs.T, fmt=fmt_bvecs, delimiter=delimiter)

    def save_image(self, relative_path):

        name = 'image.nii.gz'
        path = os.path.expanduser(relative_path)
        url = os.path.join(path, name)

        if not os.path.isdir(path):
            os.makedirs(path)

        save_nifti(url, self.image, self.affine)

        name = 'mask.nii.gz'
        path = os.path.expanduser(relative_path)
        url = os.path.join(path, name)

        if not os.path.isdir(path):
            os.makedirs(path)

        self.make_mask()

        save_nifti(url, self.mask, self.affine)

    def save_diffusion_volumes(self, relative_path):

        name = 'diffusion_volumes.nii.gz'

        path = os.path.expanduser(relative_path)
        url = os.path.join(path, name)

        if not os.path.isdir(path):
            os.makedirs(path)

        save_nifti(url, self.volumes, self.affine)

    def fit_dti(self):

        gtab = gradient_table(self.bvals, self.bvecs)

        tensor_model = dti.TensorModel(gtab, fit_method='OLS')

        print(f'\nfitting dti ... ', end='')

        tensor_fit = tensor_model.fit(self.volumes, mask=self.mask)
        print('done.')

        dti_coeffs = dti.lower_triangular(tensor_fit.quadratic_form)

        # reorder coefficients to FDT convention Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
        # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide
        self.dti = dti_coeffs[:, :, :, (0, 1, 3, 2, 4, 5)]
        self.v1 = tensor_fit.evecs[:, :, :, :, 0]
        self.evals = tensor_fit.evals

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

    def fit_odf(self):

        from dipy.io.image import load_nifti, save_nifti
        from dipy.io.gradients import read_bvals_bvecs
        from dipy.data import get_fnames

        hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
        data, affine = load_nifti(hardi_fname)
        # bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
        # gtab = gradient_table(bvals, bvecs)
        # response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
        print(f'\nfitting odf ... ', end='')

        gtab = gradient_table(self.bvals, self.bvecs)
        response, ratio = auto_response(gtab, self.volumes, roi_radius=10, fa_thr=0.7)

        # from dipy.reconst.csdeconv import recursive_response
        #
        # from dipy.reconst.dti import fractional_anisotropy
        # FA = fractional_anisotropy(self.evals)
        # MD = dti.mean_diffusivity(self.evals)
        # wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))
        #
        # gtab = gradient_table(self.bvals, self.bvecs)
        # response = recursive_response(gtab, self.volumes, mask=wm_mask, sh_order=8,
        #                               peak_thr=0.01, init_fa=0.08,
        #                               init_trace=0.0021, iter=1, convergence=0.001,
        #                               parallel=True)


        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        print('done.')

        csd_fit = csd_model.fit(self.volumes)
        self.odf = csd_fit.shm_coeff

    def save_odf(self, relative_path):

        name = 'odf.nii.gz'

        path = os.path.expanduser(relative_path)
        url = os.path.join(path, name)

        if not os.path.isdir(path):
            os.makedirs(path)

        save_nifti(url, self.odf, self.affine)


class Ellipse:

    def __init__(self, mean, covariance):
        self.covariance = covariance
        self.inv_covariance = np.linalg.pinv(self.covariance)
        self.mean = mean

    def quad_form_at(self, x):
        delta = x - self.mean
        quad_form = delta.T @ self.covariance @ delta
        return quad_form

    def exp_quad_at(self, x, sigma=1):
        return np.exp(-(1 / sigma) * self.quad_form_at(x))

    def rotate_covariance(self, angles):
        R = Rotation.from_rotvec(angles).as_matrix()
        self.covariance = R @ self.covariance @ R.T
        self.inv_covariance = R @ self.inv_covariance @ R.T


if __name__ == '__main__':
    process_diffusion_image()
