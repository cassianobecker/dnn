import os
import subprocess
import numpy as np

from dataset.hcp.hcp import HcpDiffusionDatabase
from util.logging import get_logger, set_logger
from fwk.config import Config

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.shm import SphHarmFit, sph_harm_ind_list, real_sym_sh_basis
from dipy.sims.voxel import single_tensor_odf
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model


class HcpOdfProcessor:

    def __init__(self):
        self.database = HcpDiffusionDatabase()

        self.processing_folder = os.path.expanduser(Config.config['DATABASE']['local_processing_directory'])
        if not os.path.isdir(self.processing_folder):
            os.makedirs(self.processing_folder, exist_ok=True)

        log_furl = os.path.join(self.processing_folder, 'log', 'preprocessing.log')
        if not os.path.isdir(log_furl):
            os.makedirs(os.path.join(self.processing_folder, 'log'), exist_ok=True)

        set_logger('HcpProcessor', Config.config['LOGGING']['processing_level'], log_furl)
        self.logger = get_logger('HcpProcessor')

    def process_subject(self, subject, delete_folders=False):

        self.logger.info(f'----- processing subject {subject}')

        if not os.path.exists(self._processed_tensor_url(subject)):
            self.database.get_diffusion(subject)
            odf_coeffs = self.fit_odf(subject)
            odf_coeffs = np.transpose(odf_coeffs, (3, 0, 1, 2))
            self.save_odf_tensor_image(subject, odf_coeffs)

        if delete_folders is True:
            self.database.delete_diffusion_folder(subject)
            self._delete_fsl_folder(subject)

    def save_odf_tensor_image(self, subject, odf_coeffs):

        self.logger.info(f'saving subject {subject}')

        if not os.path.isdir(self._processed_tensor_folder(subject)):
            os.makedirs(self._processed_tensor_folder(subject))
        np.savez_compressed(self._processed_tensor_url(subject), dwi_tensor=odf_coeffs)

    def fit_odf(self, subject):

        hardi_fname = self.database.url_for_file(subject, self.database.DTI_DATA)
        data, affine = load_nifti(hardi_fname)

        bval_fname = self.database.url_for_file(subject, self.database.DTI_BVALS)
        bvec_fname = self.database.url_for_file(subject, self.database.DTI_BVECS)
        bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
        gtab = gradient_table(bvals, bvecs)

        self.logger.info(f'computing auto response subject {subject}')

        response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

        self.logger.info(f'creating csd model for subject {subject}')

        csd_model = ConstrainedSphericalDeconvModel(gtab, response)

        self.logger.info(f'fitting model for subject {subject}')

        csd_fit = csd_model.fit(data)

        return csd_fit.shm_coeff

    def _processed_tensor_folder(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject)

    def _processed_tensor_url(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject, 'odf_tensor_' + subject + '.npz')

    @staticmethod
    def _filter_tensor(x, idx):
        return x[idx[0]:idx[1], idx[2]:idx[3], idx[4]:idx[5]]

    @staticmethod
    def _get_peaks(model, data):
        csd_peaks = peaks_from_model(model=model,
                                     data=data,
                                     sphere=default_sphere,
                                     relative_peak_threshold=.5,
                                     min_separation_angle=25,
                                     parallel=True)
        return csd_peaks
