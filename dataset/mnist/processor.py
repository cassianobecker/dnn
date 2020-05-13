import os

import dipy.reconst.dti as dti
import numpy as np
from dipy.core.gradients import gradient_table, generate_bvecs
from dipy.io.image import save_nifti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import auto_response

from dataset.mnist.database import MnistDatabase, KmnistDatabase
from dataset.mnist.dwi.diff_nd import successive_differences
from dataset.mnist.dwi.tube import tubify
from fwk.config import Config
from util.logging import get_logger, set_logger


class MnistProcessor:

    def __init__(self):

        mnist_dataset = Config.config['DATABASE']['mnist_dataset']
        dataset_base_path = Config.config['DATABASE']['dataset_base_path']
        local_processing_directory = os.path.join(dataset_base_path, mnist_dataset, 'processing')

        self.processing_folder = os.path.expanduser(local_processing_directory)
        if not os.path.isdir(self.processing_folder):
            os.makedirs(self.processing_folder, exist_ok=True)

        log_furl = os.path.join(self.processing_folder, 'log', 'preprocessing.log')
        if not os.path.isdir(log_furl):
            os.makedirs(os.path.join(self.processing_folder, 'log'), exist_ok=True)

        set_logger('MnistProcessor', Config.config['LOGGING']['processing_level'], log_furl)
        self.logger = get_logger('MnistProcessor')

        self.database_class_names = {
            'mnist': MnistDatabase,
            'kmnist': KmnistDatabase
        }

        self.database = self.database_class_names[mnist_dataset]()
        self.depth = int(Config.config['DATABASE']['image_depth'])

    def _path_for_idx(self, image_idx, regime):
        return os.path.join(self.processing_folder, regime, str(image_idx))

    def process_labels(self, image_idx, regime):

        self.logger.info('--- processing image {}'.format(image_idx))
        image_2d, label = self.database.get_image(image_idx, regime=regime)
        image = MnistDiffusionImageModel(image_2d, depth=self.depth)
        self.logger.info(f'saving image')
        image.save_label(self._path_for_idx(image_idx, regime), label)

    def process_image(self, image_idx, regime):

        self.logger.info('--- processing image {}'.format(image_idx))

        self.logger.info(f'generating image')
        image_2d, label = self.database.get_image(image_idx, regime=regime)
        image = MnistDiffusionImageModel(image_2d, depth=self.depth)
        self.logger.info(f'saving image')
        image.save_image(self._path_for_idx(image_idx, regime))
        image.save_label(self._path_for_idx(image_idx, regime), label)

        self.logger.info(f'generating bvals and bvecs')
        image.set_b0_image(same=True, weight=3000)
        image.generate_bvals_bvecs(n_bvals=200)
        self.logger.info(f'saving bvals and bvecs')
        image.save_bvals_bvecs(self._path_for_idx(image_idx, regime))

        self.logger.info(f'generating volumes')
        image.generate_diffusion_volumes()
        self.logger.info(f'saving volumes')
        image.save_diffusion_volumes(self._path_for_idx(image_idx, regime))

        self.logger.info(f'fitting dti')
        image.fit_dti()
        self.logger.info(f'saving dti')
        image.save_dti(self._path_for_idx(image_idx, regime))

        self.logger.info(f'fitting odf')
        image.fit_odf()
        self.logger.info(f'saving odf')
        image.save_odf(self._path_for_idx(image_idx, regime))


class MnistDiffusionImageModel:

    def __init__(self, image2d, depth=8, radius=10, bval0=100, intensity_threshold=1.e-3):

        self.image = tubify(image2d, depth)

        mask = np.zeros_like(self.image)
        mask[self.image > intensity_threshold] = 1
        self.mask = mask.astype(np.int)

        self.affine = np.eye(4)

        self.intensity_threshold = intensity_threshold

        self.b0_image = None
        self.bvals = None
        self.bvecs = None

        self.volumes = None

        self.dti = None
        self.v1 = None
        self.evals = None
        self.odf = None

        self.radius = radius
        self.bval0 = bval0

    def _diffs_at_direction_tensor(self, bval, bvec, radius=3):

        diffs = successive_differences(self.image, bvec, radius, normalize=True) + \
                successive_differences(self.image, -bvec, radius, normalize=True)

        return diffs

    @staticmethod
    def adc_fun(x):
        return 1 / (1 + np.exp(-x))

    def generate_diffusion_volumes(self):

        diffusion_volumes = []

        for i, bvec in enumerate(self.bvecs.tolist()):

            bval = self.bvals[i]
            volume = self._diffs_at_direction_tensor(bval, np.array(bvec), self.radius)

            lam_min = 2.e-4
            lam_max = 5.e-3
            adc = lam_min + volume * (lam_max - lam_min) / 1

            diffusion_volume = self.b0_image * np.exp(-bval * adc)

            diffusion_volumes.append(diffusion_volume)

        self.volumes = np.array(np.transpose(diffusion_volumes, (1, 2, 3, 0)))

    def set_b0_image(self, same=False, weight=1):

        if same is True:
            b0_image = self.image
        else:
            b0_image = np.ones_like(self.image)

        self.b0_image = weight * b0_image

    def reweight_image(self, weight):
        self.image = weight * self.image

    def make_mask(self, threshold=1e-3):
        mask = np.zeros_like(self.image)
        mask[self.image > threshold] = 1
        self.mask = mask.astype(np.int)

    def generate_bvals_bvecs(self, n_bvals=64, n_b0=8):

        bvals = self.bval0 * np.ones(n_bvals)
        bvecs = generate_bvecs(n_bvals, 1)

        if self.b0_image is not None:
            bvecs = np.concatenate((np.zeros((n_b0, 3)), bvecs), axis=0)
            bvals = np.concatenate((np.zeros(n_b0), bvals))

        self.bvals = bvals
        self.bvecs = bvecs

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

    def save_label(self, relative_path, label):

        name = 'label.nii.gz'
        path = os.path.expanduser(relative_path)
        url = os.path.join(path, name)

        if not os.path.isdir(path):
            os.makedirs(path)

        save_nifti(url, label, self.affine)

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

        tensor_fit = tensor_model.fit(self.volumes, mask=self.mask)

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

        gtab = gradient_table(self.bvals, self.bvecs)
        response, ratio = auto_response(gtab, self.volumes, roi_radius=10, fa_thr=0.7)

        csd_model = ConstrainedSphericalDeconvModel(gtab, response)

        csd_fit = csd_model.fit(self.volumes)
        self.odf = csd_fit.shm_coeff

    def save_odf(self, relative_path):

        name = 'odf.nii.gz'

        path = os.path.expanduser(relative_path)
        url = os.path.join(path, name)

        if not os.path.isdir(path):
            os.makedirs(path)

        save_nifti(url, self.odf, self.affine)
