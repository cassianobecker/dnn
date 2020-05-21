import os
import numpy as np
from dataset.hcp.covariates import Covariates
from util.logging import get_logger, set_logger
from fwk.config import Config
from util.path import absolute_path
import nibabel as nb
from util.arrays import slice_from_list_of_pairs
from dipy.io.image import load_nifti


class HcpReader:

    def __init__(self):

        self.processing_folder = os.path.expanduser(Config.config['DATABASE']['local_processing_directory'])
        if not os.path.isdir(self.processing_folder):
            os.makedirs(self.processing_folder, exist_ok=True)

        log_folder = os.path.expanduser(os.path.join(Config.config['EXPERIMENT']['results_path'], 'log'))
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder, exist_ok=True)

        log_furl = os.path.join(log_folder, 'reader.log')
        set_logger('HcpReader', Config.config['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpReader')

        self.field = Config.config['COVARIATES']['field']
        self.covariates = Covariates()

        self.model = Config.get_option('DATABASE', 'model', None)

        self.file_names = {'dti': 'dti_tensor.nii.gz', 'odf': 'odf.nii.gz'}

        if Config.config.has_option('TEMPLATE', 'mask'):
            mask_folder = absolute_path(Config.config['TEMPLATE']['folder'])
            self.mask_url = os.path.join(mask_folder, Config.config['TEMPLATE']['mask'])
        else:
            self.mask_url = None

    def load_subject_list(self, list_url, max_subjects=None):
        self.logger.info('loading subjects from ' + list_url)
        with open(absolute_path(list_url), 'r') as f:
            subjects = [s.strip() for s in f.readlines()]
            subjects = subjects[:int(max_subjects)] if max_subjects is not None else subjects

        self.logger.info('loaded ' + str(len(subjects)) + ' subjects from: ' + list_url)
        return subjects

    # def _processed_tensor_folder(self, subject):
    #     return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject)
    #
    # def _processed_tensor_url(self, subject):
    #     return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject, 'tensor_' + subject + '.npz')
    #
    # def load_dwi_tensor_image(self, subject,
    #                           region=None,
    #                           vectorize=True,
    #                           normalize=False,
    #                           mask=False,
    #                           scale=1.,
    #                           max_img_channels=None,
    #                           ):
    #
    #     try:
    #         dwi_tensor = np.load(self._processed_tensor_url(subject))['dwi_tensor']
    #     except FileNotFoundError:
    #         raise SkipSubjectException(f'File for subject {subject} not found')
    #
    #     if region is not None:
    #         slices = slice_from_list_of_pairs(region, null_offset=2)
    #         dwi_tensor = dwi_tensor[slices]
    #
    #     if self.mask_url is not None:
    #         dwi_tensor = self.apply_mask(dwi_tensor)
    #
    #     if vectorize is True and len(dwi_tensor.shape) > 4:
    #         dwi_tensor = self.vectorize_channels(dwi_tensor)
    #
    #     if normalize is True:
    #         dwi_tensor = self.normalize_channels(dwi_tensor)
    #
    #     if scale is not None:
    #         dwi_tensor = dwi_tensor * scale
    #
    #     if max_img_channels is not None:
    #         dwi_tensor = dwi_tensor[:max_img_channels, :, :, :]
    #
    #     return dwi_tensor

    def _processed_tensor_url(self, subject):
        return os.path.join(self.processing_folder, subject, 'fitted', self.file_names[self.model])

    def load_dwi_tensor_image(self, subject,
                              region=None,
                              vectorize=True,
                              normalize=False,
                              mask=False,
                              scale=1.,
                              max_img_channels=None,
                              ):

        try:
            dwi_tensor, affine = load_nifti(self._processed_tensor_url(subject))
            dwi_tensor = dwi_tensor.transpose((3, 0, 1, 2))
        except FileNotFoundError:
            raise SkipSubjectException(f'File for subject {subject} not found')

        if region is not None:
            slices = slice_from_list_of_pairs(region, null_offset=2)
            dwi_tensor = dwi_tensor[slices]

        if self.mask_url is not None:
            dwi_tensor = self.apply_mask(dwi_tensor)

        if vectorize is True and len(dwi_tensor.shape) > 4:
            dwi_tensor = self.vectorize_channels(dwi_tensor)

        if normalize is True:
            dwi_tensor = self.normalize_channels(dwi_tensor)

        if scale is not None:
            dwi_tensor = dwi_tensor * scale

        if max_img_channels is not None:
            dwi_tensor = dwi_tensor[:max_img_channels, :, :, :]

        return dwi_tensor

    def load_covariate(self, subject):
        return self.covariates.value(self.field, subject)

    def apply_mask(self, tensor):

        if self.mask_url is None:
            raise RuntimeError('No mask file set in the configuration file')

        mask_tensor = nb.load(absolute_path(self.mask_url)).get_data()
        return mask_tensor * tensor

    @staticmethod
    def vectorize_channels(tensor):

        # use FSL convention (see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide)
        # 0,0 Dxx
        # 0,1 Dxy
        # 0,2 Dxz
        # 1,1 Dyy
        # 1,2 Dyz
        # 2,2 Dzz

        if not (tensor.shape[0], tensor.shape[1]) == (3, 3):
            raise RuntimeError('Vectorization only applicable for (3, 3) DTI matrices.')

        new_tensor = np.stack((
            tensor[0, 0, :, :, :],
            tensor[0, 1, :, :, :],
            tensor[0, 2, :, :, :],
            tensor[1, 1, :, :, :],
            tensor[1, 2, :, :, :],
            tensor[2, 2, :, :, :],
        ), axis=0)

        return new_tensor

    @staticmethod
    def normalize_channels(tensor):
        einsums = {4: 'iklm->klm', 5: 'ijklm->klm'}
        einsum_str = einsums[len(tensor.shape)]

        norms = np.sqrt(np.einsum(einsum_str, tensor ** 2))
        denominator = np.average(norms, weights=norms != 0)

        return tensor/denominator

    @staticmethod
    def parse_region(region_str):
        region_str_list = [int(x.strip()) for x in region_str.split(' ')]
        return np.reshape(region_str_list, (3, 2)).tolist()


class SkipSubjectException(Exception):
    def __init(self, msg):
        super(SkipSubjectException, self).__init__(msg)
