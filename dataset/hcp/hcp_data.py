import glob
import os
import time
import subprocess
from distutils.util import strtobool

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

        self.local_folder = database_settings['DIRECTORIES']['local_server_directory']
        self.delete_nii = strtobool(database_settings['DIRECTORIES']['delete_after_downloading'])
        self.dif_downloader = DiffusionDownloader(database_settings)
        nib.imageglobals.logger = set_logger('Nibabel', database_settings['LOGGING']['nibabel_level'], log_furl)

        self.field = params['COVARIATES']['field']
        self.covariates = Covariates()

    def load_subject_list(self, list_url):

        self.logger.info('loading subjects from ' + list_url)

        with open(list_url, 'r') as f:
            subjects = [s.strip() for s in f.readlines()]

        self.logger.info('loaded ' + str(len(subjects)) + ' subjects from: ' + list_url)

        return subjects

    def process_subject(self, subject, tasks):

        self.logger.info('processing subject {}'.format(subject))
        print('getting data for subject {}'.format(subject))

        self.get_diffusion(subject)
        self.fit_dti(subject)

        dti_tensor = self.build_dti_tensor_image(subject)
        target = self.load_covariate(subject)

        return dti_tensor, target

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
            furl = os.path.join('HCP_1200', subject, 'T1w', 'Diffusion', fname)
            self.dif_downloader.load(furl, subject)
            try:
                furl = os.path.join(self.local_folder, furl)
                if 'nii' in fname:
                    dif[fname] = np.array(nib.load(furl).get_data())
                else:
                    dif[fname] = open(furl, 'r').read()
                if self.delete_nii:
                    self.dif_downloader.delete_dir(furl)
            except:
                msg = f"Error loading file {furl}."
                self.logger.error(msg)
                raise SkipSubjectException(msg)
        self.logger.debug("done")
        return dif

    def fit_dti(self, subject):

        diffusion_dir = os.path.join(self.local_folder, 'HCP_1200', subject, 'T1w', 'Diffusion')

        processed_fsl_dir = os.path.join(self.local_folder, 'HCP_1200_processed', subject, 'fsl')

        if not os.path.isdir(processed_fsl_dir):
            os.makedirs(processed_fsl_dir)

        if not os.path.isfile(os.path.join(processed_fsl_dir, 'fsl_V3.nii.gz')):
            time.sleep(10)
            dti_fit_command_str = \
                'dtifit -k {0}/data.nii.gz -o {1}/fsl -m {0}/nodif_brain_mask.nii.gz -r {0}/bvecs -b {0}/bvals ' \
                '--save_tensor'.format(diffusion_dir, processed_fsl_dir)

            pc = subprocess.run(dti_fit_command_str, shell=True, check=True)

        pass

    def build_dti_tensor_image(self, subject):
        """
        Reads in eigenvectors and eigenvalues from DTI fit and returns  3*3*i*j*k DTI array for input to nn
        """
        processed_fsl_dir = os.path.join(self.local_folder, 'HCP_1200_processed', subject, 'fsl')

        dti_tensor = 0
        for i in range(1, 4):
            evecs_file = glob.glob(os.path.join(processed_fsl_dir, '*V' + str(i) + '*'))[0]
            evals_file = glob.glob(os.path.join(processed_fsl_dir, '*L' + str(i) + '*'))[0]
            evecs = nib.load(evecs_file).get_fdata()
            evals = nib.load(evals_file).get_fdata()
            dti_tensor = dti_tensor + np.einsum('abc,abci,abcj->ijabc', evals, evecs, evecs)
        return dti_tensor

    def load_covariate(self, subject):
        return self.covariates.value(self.field, subject)


class SkipSubjectException(Exception):
    def __init(self, msg):
        super(SkipSubjectException, self).__init__(msg)
