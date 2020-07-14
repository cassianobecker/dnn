import numpy as np
import os
from os.path import join
import subprocess
import shutil

from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, recursive_response
from dipy.reconst.csdeconv import auto_response
from dipy.core.gradients import gradient_table

from fwk.config import Config
from util.path import absolute_path, copy_folder

from dataset.synth.regression import FibercupRegressionDataset

BASE_PATH_ON_CONTAINER = '/dnn/.dnn/processing'
DWI_PARAMS_FILE = 'few_param.ffp'


class SynthProcessor:

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.database_processing_path = os.path.expanduser(Config.get_option('DATABASE', 'local_processing_directory'))
        self.container_path = os.path.expanduser(Config.get_option('DWI', 'path_to_container'))
        self.container_relative_processing_path = BASE_PATH_ON_CONTAINER
        self.container_processing_path = join(self.container_path, *BASE_PATH_ON_CONTAINER.split(os.path.sep))
        self.dwi_params_path_on_container = join(self.container_processing_path, 'params')
        self.setup_dwi_params()

    def setup_dwi_params(self):
        self.copy_dwi_params()
        self.flip_evecs()

    def copy_dwi_params(self):
        src_path = join(absolute_path('dataset'), 'synth', 'dwi_params')  # resource folder in codebase
        dest_path = self.dwi_params_path_on_container

        if os.path.isdir(dest_path):
            shutil.rmtree(dest_path)

        os.makedirs(dest_path, exist_ok=True)

        suffixes = ['', '.bvals', '.bvecs']

        for suffix in suffixes:
            src = join(src_path, DWI_PARAMS_FILE + suffix)
            dest = join(dest_path, DWI_PARAMS_FILE + suffix)
            shutil.copyfile(src, dest)

    def flip_evecs(self, flips=(1, -1, 1)):
        # flip eigenvectors for compatibility between Mitk Fiberfox and FSL dtifit
        bvals_url = join(self.dwi_params_path_on_container, DWI_PARAMS_FILE + '.bvals')
        bvecs_url = join(self.dwi_params_path_on_container, DWI_PARAMS_FILE + '.bvecs')
        bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)
        new_bvecs = bvecs @ np.diag(flips)
        self.save_bvals_bvecs(bvals, new_bvecs)

    def save_bvals_bvecs(self, bvals, bvecs):
        np.savetxt(self.flipped_bvals_url(), np.expand_dims(bvals, axis=0), fmt='%d', delimiter='  ')
        np.savetxt(self.flipped_bvecs_url(), bvecs.T, fmt='%2.6f', delimiter='  ')

    def flipped_bvals_url(self):
        return join(self.dwi_params_path_on_container, 'flipped_' + DWI_PARAMS_FILE + '.bvals')

    def flipped_bvecs_url(self):
        return join(self.dwi_params_path_on_container, 'flipped_' + DWI_PARAMS_FILE + '.bvecs')

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

    def simulate_dwi(self, sample_id, relative=False):
        # setup paths and files for container use
        if relative:
            params_url = join(self.container_relative_processing_path, 'params', DWI_PARAMS_FILE)
            tracts_url = join(self.container_relative_processing_path, f'{sample_id}', 'tracts', 'tracts.fib')
            target_url = join(self.container_relative_processing_path, f'{sample_id}', 'dwi', 'data')
        else:
            params_url = join(self.make_path(None, 'params', container=True), DWI_PARAMS_FILE)
            tracts_url = join(self.make_path(sample_id, 'tracts', container=True), 'tracts.fib')
            target_url = join(self.make_path(sample_id, 'dwi', container=True), 'data')

        os.makedirs(self.make_path(sample_id, 'dwi', container=True), exist_ok=True)

        # define executables
        container_prefix = Config.get_option('DWI', 'container_prefix')
        fiberfox_executable_on_container = Config.get_option('DWI', 'fiberfox_executable_within_container')

        str_cmd = f'{container_prefix} ' \
                  f'{fiberfox_executable_on_container} ' \
                  f'-o {target_url} ' \
                  f'-i {tracts_url} ' \
                  f'-p {params_url} ' \
                  f'--verbose'

        subprocess.run(str_cmd, shell=True, check=True)

    def transfer_files_from_container(self, sample_id, delete_after=False):
        # # transfer parameters folder and do not delete source folder
        src_folder = self.dwi_params_path_on_container
        dest_folder = self.make_path(sample_id, 'params', container=False)
        copy_folder(src_path=src_folder, dest_path=dest_folder, delete_src=False)

        # transfer other folders, deletimg source folders
        folders = ['tracts', 'dwi']
        for folder in folders:
            src_folder = self.make_path(sample_id, folder, container=True)
            dest_folder = self.make_path(sample_id, folder, container=False)
            copy_folder(src_path=src_folder, dest_path=dest_folder)

        # delete folder for sample_id
        if delete_after:
            shutil.rmtree(join(self.container_processing_path, f'{sample_id}'))

    def fit_dti(self, sample_id):

        dti_params = {
            'data':  join(self.make_path(sample_id, 'dwi'), 'data.nii.gz'),
            'mask': join(self.make_path(sample_id, 'dwi'), 'data_mask.nii.gz'),
            'bvals': join(self.make_path(sample_id, 'params'), 'flipped_' + DWI_PARAMS_FILE + '.bvals'),
            'bvecs': join(self.make_path(sample_id, 'params'), 'flipped_' + DWI_PARAMS_FILE + '.bvecs'),
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

        bvals_url = join(self.make_path(sample_id, 'params'), DWI_PARAMS_FILE + '.bvals')
        bvecs_url = join(self.make_path(sample_id, 'params'), DWI_PARAMS_FILE + '.bvecs')

        bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)
        gtab = gradient_table(bvals, bvecs)

        volumes_url = join(self.make_path(sample_id, 'dwi'), 'data.nii.gz')

        volumes, volumes_affine = load_nifti(volumes_url)

        # response = recursive_response(gtab, volumes, sh_order=8,
        #                               peak_thr=0.01, init_fa=0.08,
        #                               init_trace=0.0021, iter=8, convergence=0.001,
        #                               parallel=True)
        response, ratio = auto_response(gtab, volumes, roi_center=(29, 48, 2), roi_radius=1, fa_thr=0.24)

        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        csd_fit = csd_model.fit(volumes)

        odf = csd_fit.shm_coeff
        odf_url = join(self.make_path(sample_id, 'odf'), 'odf.nii.gz')
        save_nifti(odf_url, odf, volumes_affine)
