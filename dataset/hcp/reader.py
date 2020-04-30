import os
import numpy as np
from dataset.hcp.covariates import Covariates
from util.logging import get_logger, set_logger
from fwk.config import Config
from util.path import absolute_path
import nibabel as nb
from util.arrays import slice_from_list_of_pairs


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

    def _processed_tensor_folder(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject)

    def _processed_tensor_url(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject, 'tensor_' + subject + '.npz')

    def load_dwi_tensor_image(self, subject,
                              region=None,
                              vectorize=True,
                              normalize=True,
                              mask=False,
                              max_img_channels=None,
                              ):

        try:
            dwi_tensor = np.load(self._processed_tensor_url(subject))['dwi_tensor']
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

        if max_img_channels is not None:
            dwi_tensor = dwi_tensor[:max_img_channels, :, :, :]

        return dwi_tensor

    def load_covariate(self, subject):
        return self.covariates.value(self.field, subject).argmin().astype(np.long)

    def apply_mask(self, tensor):

        if self.mask_url is None:
            raise RuntimeError('No mask file set in the configuration file')

        mask_tensor = nb.load(absolute_path(self.mask_url)).get_data()
        return mask_tensor * tensor

    @staticmethod
    def vectorize_channels(tensor):
        return tensor.reshape((tensor.shape[0]*tensor.shape[1], *tensor.shape[2:]))

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
