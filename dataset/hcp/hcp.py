import os
from distutils.util import strtobool
import shutil
import numpy as np

from dataset.hcp.downloaders import HcpDiffusionDownloader
from util.logging import get_logger, set_logger
from fwk.config import Config
from util.path import absolute_path, is_project_in_cbica


class HcpDiffusionDatabase:

    def __init__(self):

        self.processing_folder = os.path.expanduser(Config.config['DATABASE']['local_processing_directory'])
        if not os.path.isdir(self.processing_folder):
            os.makedirs(self.processing_folder, exist_ok=True)

        log_furl = os.path.join(self.processing_folder, 'log', 'preprocessing.log')
        if not os.path.isdir(log_furl):
            os.makedirs(os.path.join(self.processing_folder, 'log'), exist_ok=True)

        set_logger('HcpProcessor', Config.config['LOGGING']['processing_level'], log_furl)
        self.logger = get_logger('HcpProcessor')

        self.mirror_folder = self.get_mirror_folder()
        self.delete_nii = strtobool(Config.config['DATABASE']['delete_after_downloading'])
        self.subjects = self.load_subject_list()
        self.downloader = HcpDiffusionDownloader()

        self.DTI_BVALS = 'bvals'
        self.DTI_BVECS = 'bvecs'
        self.DTI_MASK = 'nodif_brain_mask.nii.gz'
        self.DTI_DATA = 'data.nii.gz'

    @staticmethod
    def get_mirror_folder():

        if is_project_in_cbica():
            local_server_directory = '/cbica/projects/HCP_Data_Releases'
        else:
            local_server_directory = os.path.expanduser(Config.config['DATABASE']['local_server_directory'])

        return local_server_directory

    def load_subject_list(self):

        list_url = Config.config['SUBJECTS']['process_subjects_file']

        if 'max_subjects' in Config.config['SUBJECTS'].keys():
            max_subjects = Config.config['SUBJECTS']['max_subjects']
        else:
            max_subjects = None

        self.logger.info('loading subjects from ' + list_url)
        with open(absolute_path(list_url), 'r') as f:
            subjects = [s.strip() for s in f.readlines()]
            subjects = subjects[:int(max_subjects)] if max_subjects is not None else subjects

        self.logger.info('loaded ' + str(len(subjects)) + ' subjects from: ' + list_url)
        return subjects

    def subject_batch(self, batch_index, number_of_batches):
        batch_size = int(len(self.subjects) / (number_of_batches - 1))
        initial = batch_index * batch_size
        final = (batch_index + 1) * batch_size
        return self.subjects[initial:final]

    def subject_dir(self, subject):
        return os.path.join(self.mirror_folder, 'HCP_1200', subject)

    @staticmethod
    def diffusion_dir(subject):
        return os.path.join('HCP_1200', subject, 'T1w', 'Diffusion')

    def url_for_file(self, subject, name):
        return os.path.join(self.mirror_folder, self.diffusion_dir(subject), name)

    def is_tensor_saved(self, subject):
        exists = False
        if os.path.isdir(self.processed_tensor_folder(subject)):
            exists = os.path.isfile(self.processed_tensor_url(subject))
        return exists

    def delete_diffusion_folder(self, subject):
        diffusion_dir = self.subject_dir(subject)
        if os.path.exists(diffusion_dir):
            shutil.rmtree(diffusion_dir)

    def load_dti_tensor_image(self, subject):
        dti_tensor = np.load(self.processed_tensor_url(subject))
        return dti_tensor['dti_tensor']

    def load_raw_diffusion(self, subject, fnames):
        self.logger.debug("loading diffusion for " + subject)
        dif = {}
        for fname in fnames:
            furl = os.path.join(self.diffusion_dir(subject), fname)
            self.downloader.load(furl, subject)
        self.logger.debug("done")
        return dif

    def get_diffusion(self, subject):
        fnames = [self.DTI_BVALS, self.DTI_BVECS, self.DTI_MASK, self.DTI_DATA]
        self.load_raw_diffusion(subject, fnames)
