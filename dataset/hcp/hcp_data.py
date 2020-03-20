import glob
import os
import subprocess
from distutils.util import strtobool
import shutil

import nibabel as nib
import numpy as np

from dataset.hcp.covariates import Covariates
from dataset.hcp.downloaders import DiffusionDownloader
from util.logging import get_logger, set_logger


class HcpReader:

    def __init__(self, database_settings, params):

        log_furl = os.path.join(params['FILE']['experiment_path'], 'log', 'downloader.log')
        set_logger('HcpReader', database_settings['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpReader')

        self.local_folder = database_settings['DIRECTORIES']['local_directory']

        if not os.path.isdir(self.local_folder):
            os.makedirs(self.local_folder)

        self.mirror_folder = database_settings['DIRECTORIES']['mirror_directory']

        self.delete_nii = strtobool(database_settings['DIRECTORIES']['delete_after_downloading'])
        self.dif_downloader = DiffusionDownloader(database_settings)
        nib.imageglobals.logger = set_logger('Nibabel', database_settings['LOGGING']['nibabel_level'], log_furl)

        self.field = params['COVARIATES']['field']
        self.covariates = Covariates()

        self.dti_files = {'fsl_FA.nii.gz', 'fsl_L1.nii.gz', 'fsl_L2.nii.gz', 'fsl_L3.nii.gz', 'fsl_MD.nii.gz',
                          'fsl_MO.nii.gz',  'fsl_S0.nii.gz',  'fsl_V1.nii.gz',  'fsl_V2.nii.gz',  'fsl_V3.nii.gz',
                          'fsl_tensor.nii.gz'}

    def load_subject_list(self, list_url):
        self.logger.info('loading subjects from ' + list_url)
        with open(list_url, 'r') as f:
            subjects = [s.strip() for s in f.readlines()]
        self.logger.info('loaded ' + str(len(subjects)) + ' subjects from: ' + list_url)
        return subjects

    def _subject_dir(self, subject):
        return os.path.join(self.mirror_folder, 'HCP_1200', subject)

    def _diffusion_dir(self, subject):
        return os.path.join(self.mirror_folder, 'HCP_1200', subject, 'T1w', 'Diffusion')

    def _fsl_folder(self, subject):
        return os.path.join(self.local_folder, 'HCP_1200_processed', subject, 'fsl')

    def _processed_tensor_folder(self, subject):
        return os.path.join(self.local_folder, 'HCP_1200_tensor', subject)

    def _processed_tensor_url(self, subject):
        return os.path.join(self.local_folder, 'HCP_1200_tensor', subject,  'dti_tensor_' + subject + '.npz')

    def _is_dti_processed(self, subject):
        processed_fsl_dir = self._fsl_folder(subject)
        dir_contents = os.listdir(processed_fsl_dir)
        return dir_contents == self.dti_files

    def _is_tensor_saved(self, subject):
        exists = False
        if os.path.isdir(self._processed_tensor_folder(subject)):
            exists = os.path.isfile(self._processed_tensor_url(subject))
        return exists

    def _delete_diffusion_folder(self, subject):
        diffusion_dir = self._subject_dir(subject)
        if os.path.exists(diffusion_dir):
            shutil.rmtree(diffusion_dir)

    def _delete_fsl_folder(self, subject):
        processed_fsl_dir = self._fsl_folder(subject)
        if os.path.exists(processed_fsl_dir):
            shutil.rmtree(processed_fsl_dir)

    def process_subject(self, subject, delete_folders=False):

        self.logger.info('processing subject {}'.format(subject))

        if not os.path.exists(self._processed_tensor_url(subject)):
            self.get_diffusion(subject)
            self.fit_dti(subject)
            self.save_dti_tensor_image(subject)

        if delete_folders is True:
            self._delete_diffusion_folder(subject)
            self._delete_fsl_folder(subject)

    def save_dti_tensor_image(self, subject):
        if not os.path.isdir(self._processed_tensor_folder(subject)):
            os.makedirs(self._processed_tensor_folder(subject))
        dti_tensor = self.build_dti_tensor_image(subject)
        np.savez_compressed(self._processed_tensor_url(subject), dti_tensor=dti_tensor)

    def load_dti_tensor_image(self, subject):
        dti_tensor = np.load(self._processed_tensor_url(subject))
        return dti_tensor['dti_tensor']

    # ##################### DIFFUSION #################################

    def get_diffusion(self, subject):

        DTI_BVALS = 'bvals'
        DTI_BVECS = 'bvecs'
        DTI_MASK = 'nodif_brain_mask.nii.gz'
        DTI_DATA = 'data.nii.gz'

        fnames = [DTI_BVALS, DTI_BVECS, DTI_MASK, DTI_DATA]

        self.load_raw_diffusion(subject, fnames)

    def load_raw_diffusion(self, subject, fnames):
        self.logger.debug("loading diffusion for " + subject)
        dif = {}
        for fname in fnames:
            furl = os.path.join(self._diffusion_dir(subject), fname)
            self.dif_downloader.load(furl, subject)
        self.logger.debug("done")
        return dif

    def fit_dti(self, subject):

        diffusion_dir = self._diffusion_dir(subject)
        processed_fsl_dir = self._fsl_folder(subject)

        if os.path.isdir(processed_fsl_dir):
            dti_file = set(os.listdir(processed_fsl_dir))
        else:
            dti_file = {}

        if dti_file == self.dti_files:
            self.logger.info('processed dti files found for subject {}'.format(subject))

        else:
            self.logger.info('processing dti files for subject {}'.format(subject))
            if not os.path.isdir(processed_fsl_dir):
                os.makedirs(processed_fsl_dir)

            dti_fit_command_str = \
                'dtifit -k {0}/data.nii.gz -o {1}/fsl -m {0}/nodif_brain_mask.nii.gz -r {0}/bvecs -b {0}/bvals ' \
                '--save_tensor'.format(diffusion_dir, processed_fsl_dir)

            subprocess.run(dti_fit_command_str, shell=True, check=True)

    def build_dti_tensor_image(self, subject):
        """
        Reads in eigenvectors and eigenvalues from DTI fit and returns  3*3*i*j*k DTI array for input to nn
        """
        processed_fsl_dir = self._fsl_folder(subject)

        dti_tensor = 0
        for i in range(1, 4):
            evecs_file = glob.glob(os.path.join(processed_fsl_dir, '*V' + str(i) + '*'))[0]
            evals_file = glob.glob(os.path.join(processed_fsl_dir, '*L' + str(i) + '*'))[0]
            evecs = nib.load(evecs_file).get_fdata()
            evals = nib.load(evals_file).get_fdata()
            dti_tensor = dti_tensor + np.einsum('abc,abci,abcj->ijabc', evals, evecs, evecs)
        return dti_tensor

    def load_covariate(self, subject):
        return self.covariates.value(self.field, subject).argmin()


class SkipSubjectException(Exception):
    def __init(self, msg):
        super(SkipSubjectException, self).__init__(msg)
