import os
import subprocess
import shutil
import numpy as np

from util.logging import get_logger, set_logger
from fwk.config import Config
from util.path import absolute_path, is_project_in_cbica

from dataset.hcp.hcp import HcpDiffusionDatabase
from dataset.hcp.dwi.registration import find_prealign_affine, find_warp, apply_warp_on_volumes

from dipy.io.image import load_nifti, save_nifti


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
        self.mask_file = Config.get_option('TEMPLATE', 'mask', 'FMRIB58_FA-mask_125mm.nii.gz')

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

        warped_data_url = self._url_registered_dwi(subject, 'warped_fa.nii.gz')
        save_nifti(warped_data_url, warped_fa, warped_affine)

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

    def _get_moving_fa(self, subject):

        dti_params = {
            'data': self._url_mirror_dwi(subject, 'data.nii.gz'),
            'mask': self._url_mirror_dwi(subject, 'nodif_brain_mask.nii.gz'),
            'bvals': self._url_mirror_dwi(subject, 'bvals'),
            'bvecs': self._url_mirror_dwi(subject, 'bvecs'),
            'output': self._url_moving_dwi(subject, 'dti')
        }

        self._perform_dti_fit(dti_params, save_tensor=False)

        moving_FA_url = self._url_moving_dwi(subject, 'dti_FA.nii.gz')

        return moving_FA_url

    def _perform_dti_fit(self, dti_params, save_tensor=False):

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
            'output': self._url_registered_dwi(subject, 'dti')
        }

        self._perform_dti_fit(dti_params, save_tensor=False)

    def fit_odf(self, subject):
        pass

    def _process_subject(self, subject, delete_folders=False):
        self.register(subject)
        self.fit_dti(subject)
        self.fit_odf(subject)

    def process_subject(self, subject, delete_folders=False):

        self.logger.info('processing subject {}'.format(subject))

        if not os.path.exists(self._processed_tensor_url(subject)):
            self.database.get_diffusion(subject)

        self._process_subject(subject, delete_folders)

        # if delete_folders is True:
        #     self.database.delete_diffusion_folder(subject)
        #
        #     self._delete_fsl_folder(subject)
        #     self._delete_conversion_folder(subject)
        #     self._delete_reg_folder(subject)
        #     self._delete_ants_folder(subject)

    # @staticmethod
    # def _dti_files():
    #     return {'fsl_FA.nii.gz', 'fsl_L1.nii.gz', 'fsl_L2.nii.gz', 'fsl_L3.nii.gz', 'fsl_MD.nii.gz',
    #             'fsl_MO.nii.gz',  'fsl_S0.nii.gz',  'fsl_V1.nii.gz',  'fsl_V2.nii.gz',  'fsl_V3.nii.gz',
    #             'fsl_tensor.nii.gz'}
    #
    # def _fsl_folder(self, subject):
    #     return os.path.join(self.processing_folder, 'HCP_1200_processed', subject, 'fsl')

    def _processed_tensor_folder(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject)

    def _processed_tensor_url(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject, 'dti_tensor_' + subject + '.npz')

    # def _is_dti_processed(self, subject):
    #     ants_dir = self._ants_folder(subject)
    #     dir_contents = os.listdir(ants_dir)
    #     return dir_contents == self.ants_dti_files
    #
    # def _is_tensor_saved(self, subject):
    #     exists = False
    #     if os.path.isdir(self._processed_tensor_folder(subject)):
    #         exists = os.path.isfile(self._processed_tensor_url(subject))
    #     return exists
    #
    # def _delete_fsl_folder(self, subject):
    #     processed_fsl_dir = self._fsl_folder(subject)
    #     if os.path.exists(processed_fsl_dir):
    #         shutil.rmtree(processed_fsl_dir)

    def save_dti_tensor_image(self, subject):
        if not os.path.isdir(self._processed_tensor_folder(subject)):
            os.makedirs(self._processed_tensor_folder(subject))
        dti_tensor = self.build_dti_tensor_image(subject)
        np.savez_compressed(self._processed_tensor_url(subject), dwi_tensor=dti_tensor)

    # def fit_dti(self, subject):
    #
    #     diffusion_dir = os.path.join(self.database.mirror_folder, self.database.diffusion_dir(subject))
    #     processed_fsl_dir = self._fsl_folder(subject)
    #
    #     if os.path.isdir(processed_fsl_dir):
    #         dti_file = set(os.listdir(processed_fsl_dir))
    #     else:
    #         dti_file = {}
    #
    #     if dti_file == self.dti_files:
    #         self.logger.info('processed dti files found for subject {}'.format(subject))
    #
    #     else:
    #         self.logger.info('processing dti files for subject {}'.format(subject))
    #         if not os.path.isdir(processed_fsl_dir):
    #             os.makedirs(processed_fsl_dir)
    #
    #         dti_fit_command_str = \
    #             'dtifit -k {0}/data.nii.gz -o {1}/fsl -m {0}/nodif_brain_mask.nii.gz -r {0}/bvecs -b {0}/bvals ' \
    #             '--save_tensor'.format(diffusion_dir, processed_fsl_dir)
    #
    #         subprocess.run(dti_fit_command_str, shell=True, check=True)

