import numpy as np
import os
from os.path import join
import subprocess
import shutil

from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import auto_response
from dipy.core.gradients import gradient_table

from fwk.config import Config
from util.path import absolute_path, copy_folder

from dataset.synth.regression import FibercupRegressionDataset


class SynthProcessor:

    def __init__(self, dry_run=False):

        self.dry_run = dry_run
        self.database_processing_path = os.path.expanduser(Config.get_option('DATABASE', 'local_processing_directory'))

        self.container_path = os.path.expanduser(Config.get_option('DWI', 'path_to_container'))
        self.container_rel_proc_path = Config.get_option('DWI', 'container_relative_processing_path')
        self.container_processing_path = join(self.container_path, *self.container_rel_proc_path.split(os.path.sep))

        self.dwi_params_file = (Config.get_option('DWI', 'dwi_params_file'))

    def make_path(self, sample_id, paths, container=False):
        if container:
            path = join(self.container_processing_path, f'{sample_id}', paths)
        else:
            path = join(self.database_processing_path, f'{sample_id}', paths)

        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def process_subject(self, sample_id):
        self.create_tractogram_files(sample_id)
        self.setup_dwi_params(sample_id)
        self.simulate_dwi(sample_id)
        self.transfer_files_from_container(sample_id, delete_after=True)
        self.fit_dti(sample_id)
        self.fit_odf(sample_id)

    def create_tractogram_files(self, sample_id):
        regression_dataset = FibercupRegressionDataset()
        tractogram, covariate = regression_dataset.generate_tractogram_and_covariate()
        tract_path = self.make_path(sample_id, 'tracts', container=True)
        regression_dataset.save_tract_and_label(tract_path, tractogram, label=covariate)
        mask_path = self.make_path(sample_id, 'dwi', container=True)
        regression_dataset.make_mask(mask_path)

    def setup_dwi_params(self, sample_id):
        self._copy_dwi_params(sample_id)
        self._flip_evecs(sample_id, flips=(1, -1, 1))

    def _copy_dwi_params(self, sample_id):
        src_path = join(absolute_path('dataset'), 'synth', 'dwi_params')  # resource folder in codebase
        dest_path = self.make_path(sample_id, 'params', container=True)
        os.makedirs(dest_path, exist_ok=True)

        suffixes = ['', '.bvals', '.bvecs']
        for suffix in suffixes:
            src = join(src_path, self.dwi_params_file + suffix)
            dest = join(dest_path, self.dwi_params_file + suffix)
            shutil.copyfile(src, dest)

    def _flip_evecs(self, sample_id, flips=(1, -1, 1)):
        # flip eigenvectors for compatibility between Mitk Fiberfox and FSL dtifit
        bvals_url = join(self.make_path(sample_id, 'params', container=True), self.dwi_params_file + '.bvals')
        bvecs_url = join(self.make_path(sample_id, 'params', container=True), self.dwi_params_file + '.bvecs')

        bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)
        new_bvecs = bvecs @ np.diag(flips)

        flipped_bvals_url = join(self.make_path(sample_id, 'params', container=True),
                                 'flipped_' + self.dwi_params_file + '.bvals')
        np.savetxt(flipped_bvals_url, np.expand_dims(bvals, axis=0), fmt='%d', delimiter='  ')

        flipped_bvecs_url = join(self.make_path(sample_id, 'params', container=True),
                                 'flipped_' + self.dwi_params_file + '.bvecs')
        np.savetxt(flipped_bvecs_url, new_bvecs.T, fmt='%2.6f', delimiter='  ')

    def simulate_dwi(self, sample_id):
        # setup paths and files for container use

        container = Config.get_option('DWI', 'container_type', 'docker')

        if container == 'docker':
            params_url = join(self.container_rel_proc_path, f'{sample_id}', 'params', self.dwi_params_file)
            tracts_url = join(self.container_rel_proc_path, f'{sample_id}', 'tracts', 'tracts.fib')
            target_url = join(self.container_rel_proc_path, f'{sample_id}', 'dwi', 'data')
            container_prefix = Config.get_option('DWI', 'docker_container_prefix')
            fiberfox_executable = Config.get_option('DWI', 'fiberfox_executable_within_container')

        elif container == 'singularity':
            params_url = join(self.make_path(sample_id, 'params', container=True), self.dwi_params_file)
            tracts_url = join(self.make_path(sample_id, 'tracts', container=True), 'tracts.fib')
            target_url = join(self.make_path(sample_id, 'dwi', container=True), 'data')
            container_prefix = Config.get_option('DWI', 'singularity_container_prefix')
            fiberfox_executable = os.path.expanduser(join(
                self.container_path,
                *Config.get_option('DWI', 'fiberfox_executable_within_container').split(os.path.sep),
            ))

        os.makedirs(self.make_path(sample_id, 'dwi', container=True), exist_ok=True)

        str_cmd = f'{container_prefix} ' \
                  f'{fiberfox_executable} ' \
                  f'-o {target_url} ' \
                  f'-i {tracts_url} ' \
                  f'-p {params_url} ' \
                  f'--verbose'

        subprocess.run(str_cmd, shell=True, check=True)

    def transfer_files_from_container(self, sample_id, delete_after=False):

        folders = ['tracts', 'dwi', 'params']
        for folder in folders:
            src_folder = self.make_path(sample_id, folder, container=True)
            dest_folder = self.make_path(sample_id, folder, container=False)
            copy_folder(src_path=src_folder, dest_path=dest_folder)

        # delete folder for sample_id
        if delete_after:
            shutil.rmtree(join(self.container_processing_path, f'{sample_id}'))

    def fit_dti(self, sample_id):

        dti_params = {
            'data': join(self.make_path(sample_id, 'dwi'), 'data.nii.gz'),
            'mask': join(self.make_path(sample_id, 'dwi'), 'data_mask.nii.gz'),
            'bvals': join(self.make_path(sample_id, 'params'), 'flipped_' + self.dwi_params_file + '.bvals'),
            'bvecs': join(self.make_path(sample_id, 'params'), 'flipped_' + self.dwi_params_file + '.bvecs'),
            'output': join(self.make_path(sample_id, 'dti'), 'dti'),
        }

        self._perform_dti_fit(dti_params, save_tensor=True)

        # convert file for compatibility on CBICA for older versions of FSL
        dti_tensor_url = join(self.make_path(sample_id, 'dti'), 'dti_tensor.*')
        fslconvert_command_str = f'fslchfiletype NIFTI_GZ {dti_tensor_url}'
        subprocess.run(fslconvert_command_str, shell=True, check=True)

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

    def fit_odf(self, sample_id):

        bvals_url = join(self.make_path(sample_id, 'params'), self.dwi_params_file + '.bvals')
        bvecs_url = join(self.make_path(sample_id, 'params'), self.dwi_params_file + '.bvecs')

        bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)
        gtab = gradient_table(bvals, bvecs)

        volumes_url = join(self.make_path(sample_id, 'dwi'), 'data.nii.gz')

        volumes, volumes_affine = load_nifti(volumes_url)

        response, ratio = auto_response(gtab, volumes, roi_center=(29, 48, 2), roi_radius=1, fa_thr=0.24)

        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        csd_fit = csd_model.fit(volumes)

        odf = csd_fit.shm_coeff

        mask, mask_affine = load_nifti(join(self.make_path(sample_id, 'dwi'), 'data_mask.nii.gz'))
        masked_odf = (mask[..., 0] * odf.transpose((2, 3, 1, 0))).transpose((3, 2, 0, 1))

        odf_url = join(self.make_path(sample_id, 'odf'), 'odf.nii.gz')
        save_nifti(odf_url, masked_odf, volumes_affine)
