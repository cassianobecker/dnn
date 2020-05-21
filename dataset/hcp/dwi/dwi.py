import os
import subprocess
import shutil

from util.logging import get_logger, set_logger
from fwk.config import Config
from util.path import absolute_path

from dataset.hcp.hcp import HcpDiffusionDatabase
from dataset.hcp.dwi.registration import find_prealign_affine, find_warp, apply_warp_on_volumes

from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import auto_response
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs


class HcpDwiProcessor:

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

        self.template_folder = absolute_path(Config.config['TEMPLATE']['folder'])
        self.template_file = Config.get_option('TEMPLATE', 'template', 'FMRIB58_FA_125mm.nii.gz')

        # instead of 'FMRIB58_FA-mask_125mm.nii.gz'
        self.mask_file = Config.get_option('TEMPLATE', 'mask', 'FMRIB58_FA-mask_125mm_edit.nii.gz')

    def dti_fit_moving(self, subject):

        dti_params = {
            'data': self._url_mirror_dwi(subject, 'data.nii.gz'),
            'mask': self._url_mirror_dwi(subject, 'nodif_brain_mask.nii.gz'),
            'bvals': self._url_mirror_dwi(subject, 'bvals'),
            'bvecs': self._url_mirror_dwi(subject, 'bvecs'),
            'output': self._url_moving_dwi(subject, 'dti')
        }

        self._perform_dti_fit(dti_params, save_tensor=True)

    def process_subject(self, subject, delete_folders=False):

        self.logger.info(f'processing subject {subject}')

        self.logger.info(f'registering subject {subject}')
        self.register(subject)

        self.logger.info(f'fitting dti for subject {subject}')
        self.fit_dti(subject)

        self.logger.info(f'fitting odf for subject {subject}')
        self.fit_odf(subject)

        if delete_folders is True:
            self.logger.info(f'cleaning up for subject {subject}')
            self.clean(subject)

    def register(self, subject):

        static_url = self._template_fa_url()
        static, static_affine = load_nifti(static_url)

        moving_url = self._get_moving_fa(subject)

        moving, moving_affine = load_nifti(moving_url)

        # find linear registration transformation
        pre_align_affine = find_prealign_affine(static, static_affine,  moving, moving_affine)

        # find nonlinear registration transformation
        mapping = find_warp(pre_align_affine, static, static_affine,  moving, moving_affine)

        # apply warp on moving fa image
        warped_fa = mapping.transform(moving)

        moving_volumes_url = self._url_mirror_dwi(subject, 'data.nii.gz')
        moving_volumes, moving_affine = load_nifti(moving_volumes_url)

        warped_data = apply_warp_on_volumes(moving_volumes, mapping)

        template_mask_url = self._template_mask_url()
        template_mask, template_affine = load_nifti(template_mask_url)

        self._save_registered_files(subject, warped_data, warped_fa, template_mask, template_affine)

    def _save_registered_files(self, subject, warped_data, warped_fa, warped_mask, warped_affine):

        warped_fa_url = self._url_registered_dwi(subject, 'warped_fa.nii.gz')
        save_nifti(warped_fa_url, warped_fa, warped_affine)

        warped_masked_fa = warped_fa * warped_mask
        warped_masked_fa_url = self._url_registered_dwi(subject, 'warped_masked_fa.nii.gz')
        save_nifti(warped_masked_fa_url, warped_masked_fa, warped_affine)

        warped_mask_url = self._url_registered_dwi(subject, 'nodif_brain_mask.nii.gz')
        save_nifti(warped_mask_url, warped_mask, warped_affine)

        warped_data_url = self._url_registered_dwi(subject, 'data.nii.gz')
        save_nifti(warped_data_url, warped_data, warped_affine)

        bvals_url = self._url_mirror_dwi(subject, 'bvals')
        warped_bvals_url = self._url_registered_dwi(subject, 'bvals')
        shutil.copyfile(bvals_url, warped_bvals_url)

        bvecs_url = self._url_mirror_dwi(subject, 'bvecs')
        warped_bvecs_url = self._url_registered_dwi(subject, 'bvecs')
        shutil.copyfile(bvecs_url, warped_bvecs_url)

    def _get_moving_fa(self, subject):

        dti_params = {
            'data': self._url_mirror_dwi(subject, 'data.nii.gz'),
            'mask': self._url_mirror_dwi(subject, 'nodif_brain_mask.nii.gz'),
            'bvals': self._url_mirror_dwi(subject, 'bvals'),
            'bvecs': self._url_mirror_dwi(subject, 'bvecs'),
            'output': self._url_moving_dwi(subject, 'dti')
        }

        self._perform_dti_fit(dti_params, save_tensor=False)

        moving_fa_url = self._url_moving_dwi(subject, 'dti_FA.*')
        fslconvert_command_str = f'fslchfiletype NIFTI_GZ {moving_fa_url}'
        subprocess.run(fslconvert_command_str, shell=True, check=True)

        converted_moving_fa_url = self._url_moving_dwi(subject, 'dti_FA.nii.gz')

        return converted_moving_fa_url

    @staticmethod
    def _perform_dti_fit(dti_params, save_tensor=False):

        dti_fit_command_str = f"dtifit " \
                              f"-k {dti_params['data']} " \
                              f"-o {dti_params['output']} " \
                              f"-m {dti_params['mask']} " \
                              f"-r {dti_params['bvecs']} " \
                              f"-b {dti_params['bvals']} "

        if save_tensor is True:
            dti_fit_command_str += '--save_tensor'

        subprocess.run(dti_fit_command_str, shell=True, check=True)

    def fit_dti(self, subject):

        dti_params = {
            'data': self._url_registered_dwi(subject, 'data.nii.gz'),
            'mask': self._url_registered_dwi(subject, 'nodif_brain_mask.nii.gz'),
            'bvals': self._url_registered_dwi(subject, 'bvals'),
            'bvecs': self._url_registered_dwi(subject, 'bvecs'),
            'output': self._url_fitted_dwi(subject, 'dti')
        }

        self._perform_dti_fit(dti_params, save_tensor=True)

        registered_tensor_url = self._url_fitted_dwi(subject, 'dti_tensor.*')
        fslconvert_command_str = f'fslchfiletype NIFTI_GZ {registered_tensor_url}'
        subprocess.run(fslconvert_command_str, shell=True, check=True)

    def fit_odf(self, subject):

        bvals_url = self._url_registered_dwi(subject, 'bvals')
        bvecs_url = self._url_registered_dwi(subject, 'bvecs')
        volumes_url = self._url_registered_dwi(subject, 'data.nii.gz')

        bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)
        gtab = gradient_table(bvals, bvecs)

        volumes, volumes_affine = load_nifti(volumes_url)

        response, ratio = auto_response(gtab,  volumes, roi_radius=10, fa_thr=0.7)

        csd_model = ConstrainedSphericalDeconvModel(gtab, response)

        csd_fit = csd_model.fit(volumes)

        odf = csd_fit.shm_coeff

        mask_url = self._url_registered_dwi(subject, 'nodif_brain_mask.nii.gz')

        mask, affine = load_nifti(mask_url)

        # apply mask by transposing tensor dimensions
        odf_masked = (odf.transpose((3, 0, 1, 2)) * mask).transpose((1, 2, 3, 0))

        odf_url = self._url_fitted_dwi(subject, 'odf.nii.gz')

        save_nifti(odf_url, odf_masked, volumes_affine)

    def clean(self, subject):
        data_url = self._url_registered_dwi(subject, 'data.nii.gz')
        os.remove(data_url)

    def _template_fa_url(self):
        return os.path.join(self.template_folder, self.template_file)

    def _template_mask_url(self):
        return os.path.join(self.template_folder, self.mask_file)

    def _url_mirror_dwi(self, subject, file_name):
        return os.path.join(self.database.get_mirror_folder(), 'HCP_1200', subject, 'T1w', 'Diffusion', file_name)

    def _path_moving_dwi(self, subject):
        path = os.path.join(self.processing_folder, subject, 'moving')
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def _url_moving_dwi(self, subject, file_name):
        return os.path.join(self._path_moving_dwi(subject), file_name)

    def _path_registered_dwi(self, subject):
        path = os.path.join(self.processing_folder, subject, 'registered')
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def _url_registered_dwi(self, subject, file_name):
        return os.path.join(self._path_registered_dwi(subject), file_name)

    def _path_fitted_dwi(self, subject):
        path = os.path.join(self.processing_folder, subject, 'fitted')
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def _url_fitted_dwi(self, subject, file_name):
        return os.path.join(self._path_fitted_dwi(subject), file_name)
