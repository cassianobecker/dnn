import os
import numpy as np
from dataset.hcp.covariates import Covariates
from util.logging import get_logger, set_logger
from fwk.config import Config
from util.path import absolute_path
import nibabel as nb
from util.arrays import slice_from_list_of_pairs
from util.string import list_from


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
            self.mask_file = os.path.join(mask_folder, Config.config['TEMPLATE']['mask'])
        else:
            self.mask_file = None

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
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject, 'dti_tensor_' + subject + '.npz')

    def load_dti_tensor_image(self, subject, region=None):
        dti_tensor = np.load(self._processed_tensor_url(subject))['dti_tensor']
        if region is not None:
            slices = slice_from_list_of_pairs(region, null_offset=2)
            dti_tensor = dti_tensor[slices]
        return dti_tensor

    def load_covariate(self, subject):
        return self.covariates.value(self.field, subject).argmin()

    def apply_mask(self, tensor):

        if self.mask_file is None:
            raise RuntimeError('No mask file set in the configuration file')

        mask_tensor = nb.load(absolute_path(self.mask_file)).get_data()
        return mask_tensor * tensor

    def parse_region(self, region_str):
        region_str_list = [int(x.strip()) for x in region_str.split(' ')]
        return np.reshape(region_str_list, (3, 2)).tolist()


    def get_tensor_size(self):
        size = self.load_dti_tensor_image()
        return size

class SkipSubjectException(Exception):
    def __init(self, msg):
        super(SkipSubjectException, self).__init__(msg)
