import os
import numpy as np
from dataset.hcp.covariates import Covariates
from util.logging import get_logger, set_logger
from fwk.config import Config
from util.path import absolute_path


class HcpReader:

    def __init__(self):

        log_furl = os.path.join(Config.config['EXPERIMENT']['results_path'], 'log', 'downloader.log')
        set_logger('HcpReader', Config.config['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpReader')

        self.processing_folder = os.path.expanduser(Config.config['DATABASE']['local_processing_directory'])
        self.field = Config.config['COVARIATES']['field']
        self.covariates = Covariates()

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

    def load_dti_tensor_image(self, subject):
        dti_tensor = np.load(self._processed_tensor_url(subject))
        return dti_tensor['dti_tensor']

    def load_covariate(self, subject):
        return self.covariates.value(self.field, subject).argmin()


class SkipSubjectException(Exception):
    def __init(self, msg):
        super(SkipSubjectException, self).__init__(msg)
