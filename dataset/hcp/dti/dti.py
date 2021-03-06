import glob
import os
import subprocess
import shutil

import nibabel as nib
import numpy as np

from util.logging import get_logger, set_logger
from fwk.config import Config
from util.path import absolute_path, is_project_in_cbica

from dataset.hcp.hcp import HcpDiffusionDatabase


class HcpDtiProcessor:

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

        nib.imageglobals.logger = set_logger('Nibabel', Config.config['LOGGING']['nibabel_level'], log_furl)

        self.template_folder = absolute_path(Config.config['TEMPLATE']['folder'])
        self.template_file = Config.config['TEMPLATE']['template']

        self.dti_files = self._dti_files()
        self.converted_dti_files = self._converted_diti_files()
        self.registered_dti_files = self._registered_dti_files()
        self.ants_dti_files = self._converted_diti_files()

    def process_subject(self, subject, delete_folders=False):

        self.logger.info('processing subject {}'.format(subject))

        if not os.path.exists(self._processed_tensor_url(subject)):
            self.database.get_diffusion(subject)

            self.fit_dti(subject)
            self.convert_dti(subject)
            self.register_dti(subject)
            self.get_eigen(subject)
            self.save_dti_tensor_image(subject)

        if delete_folders is True:
            self.database.delete_diffusion_folder(subject)

            self._delete_fsl_folder(subject)
            self._delete_conversion_folder(subject)
            self._delete_reg_folder(subject)
            self._delete_ants_folder(subject)

    @staticmethod
    def _dti_files():
        return {'fsl_FA.nii.gz', 'fsl_L1.nii.gz', 'fsl_L2.nii.gz', 'fsl_L3.nii.gz', 'fsl_MD.nii.gz',
                'fsl_MO.nii.gz',  'fsl_S0.nii.gz',  'fsl_V1.nii.gz',  'fsl_V2.nii.gz',  'fsl_V3.nii.gz',
                'fsl_tensor.nii.gz'}

    @staticmethod
    def _converted_diti_files():
        return  {'comp_dxz.nii.gz', 'FA.nii.gz', 'comp_dxx.nii.gz',
                 'comp_dyz.nii.gz', 'comp_dzz.nii.gz', 'comp_dyy.nii.gz',
                 'comp_dxy.nii.gz', 'dtUpper.nii.gz', 'DT.nii.gz'}

    @staticmethod
    def _registered_dti_files():
        return {'DTDeformed.nii.gz', 'FA_reg_0GenericAffine.mat', 'FA_reg_InverseWarped.nii.gz',
                'FA_reg_combinedWarp.nii.gz', 'FA_reg_Warped.nii.gz', 'DTReorientedWarp.nii.gz',
                'FA_reg_1InverseWarp.nii.gz', 'FA_reg_1Warp.nii.gz'}

    @staticmethod
    def _ants_dti_files():
        return {'V1Deformed.nii.gz', 'L1Deformed.nii.gz', 'L3Deformed.nii.gz',
                'V3Deformed.nii.gz', 'V2Deformed.nii.gz', 'L2Deformed.nii.gz'}

    def _fsl_folder(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_processed', subject, 'fsl')

    def _conversion_folder(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_processed', subject, 'converted')

    def _reg_folder(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_processed', subject, 'reg')

    def _ants_folder(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_processed', subject, 'ants')

    def _processed_tensor_folder(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject)

    def _processed_tensor_url(self, subject):
        return os.path.join(self.processing_folder, 'HCP_1200_tensor', subject, 'dti_tensor_' + subject + '.npz')

    def _is_dti_processed(self, subject):
        ants_dir = self._ants_folder(subject)
        dir_contents = os.listdir(ants_dir)
        return dir_contents == self.ants_dti_files

    def _is_tensor_saved(self, subject):
        exists = False
        if os.path.isdir(self._processed_tensor_folder(subject)):
            exists = os.path.isfile(self._processed_tensor_url(subject))
        return exists

    def _delete_fsl_folder(self, subject):
        processed_fsl_dir = self._fsl_folder(subject)
        if os.path.exists(processed_fsl_dir):
            shutil.rmtree(processed_fsl_dir)

    def _delete_conversion_folder(self, subject):
        converted_dir = self._conversion_folder(subject)
        if os.path.exists(converted_dir):
            shutil.rmtree(converted_dir)

    def _delete_reg_folder(self, subject):
        registered_dti_dir = self._reg_folder(subject)
        if os.path.exists(registered_dti_dir):
            shutil.rmtree(registered_dti_dir)

    def _delete_ants_folder(self, subject):
        ants_dir = self._ants_folder(subject)
        if os.path.exists(ants_dir):
            shutil.rmtree(ants_dir)

    def save_dti_tensor_image(self, subject):
        if not os.path.isdir(self._processed_tensor_folder(subject)):
            os.makedirs(self._processed_tensor_folder(subject))
        dti_tensor = self.build_dti_tensor_image(subject)
        np.savez_compressed(self._processed_tensor_url(subject), dwi_tensor=dti_tensor)

    def _is_dti_processed(self, subject):

        processed_fsl_dir = self._fsl_folder(subject)

        if os.path.isdir(processed_fsl_dir):
            dti_file = set(os.listdir(processed_fsl_dir))
        else:
            dti_file = {}

        return dti_file == self.dti_files

        diffusion_dir = os.path.join(self.database.mirror_folder, self.database.diffusion_dir(subject))
        processed_fsl_dir = self._fsl_folder(subject)

        if self._is_dti_processed(self, subject):
            self.logger.info('processed dti files found for subject {}'.format(subject))
        else:
            self.logger.info('processing dti files for subject {}'.format(subject))
            if not os.path.isdir(processed_fsl_dir):
                os.makedirs(processed_fsl_dir)

            dti_fit_command_str = \
                'dtifit -k {0}/data.nii.gz -o {1}/fsl -m {0}/nodif_brain_mask.nii.gz -r {0}/bvecs -b {0}/bvals ' \
                '--save_tensor'.format(diffusion_dir, processed_fsl_dir)

            subprocess.run(dti_fit_command_str, shell=True, check=True)

    def fit_dti(self, subject):

        diffusion_dir = os.path.join(self.database.mirror_folder, self.database.diffusion_dir(subject))
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

    def convert_dti(self, subject):
        """
        Reads in FSL DTI outputs and convert to be ANTs-friendly 
        """        
        processed_fsl_dir = self._fsl_folder(subject)
        converted_dir = self._conversion_folder(subject)

        if os.path.isdir(converted_dir):
            converted_file = set(os.listdir(converted_dir))
        else:
            converted_file = {}

        if converted_file == self.converted_dti_files:
            self.logger.info('converted dti files found for subject {}'.format(subject))
        else:
            self.logger.info('converting dti files for subject {}'.format(subject))
            if not os.path.isdir(converted_dir):
                os.makedirs(converted_dir)

            if os.path.exists(os.path.join(processed_fsl_dir, 'fsl_tensor.hdr')):

                fslconvert_command_str = 'fslchfiletype NIFTI_GZ {0}/fsl_tensor.*'.format(processed_fsl_dir)
                subprocess.run(fslconvert_command_str, shell=True, check=True)

            ants_command_str = \
                'ImageMath 3 {1}/dtUpper.nii.gz 4DTensorTo3DTensor {0}/fsl_tensor.nii.gz' \
                .format(processed_fsl_dir, converted_dir)
            subprocess.run(ants_command_str, shell=True, check=True)

            comps = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
            for i in range(len(comps)):
                ants_command_str = \
                    'ImageMath 3 {0}/comp_d{1}.nii.gz TensorToVectorComponent {0}/dtUpper.nii.gz {2}' \
                    .format(converted_dir, comps[i], i+3)
                subprocess.run(ants_command_str, shell=True, check=True)

            ants_command_str = \
                'ImageMath 3 {0}/DT.nii.gz ComponentTo3DTensor {0}/comp_d .nii.gz' \
                .format(converted_dir)
            subprocess.run(ants_command_str, shell=True, check=True)

            ants_command_str = \
                'ImageMath 3 {0}/FA.nii.gz TensorFA {0}/DT.nii.gz' \
                .format(converted_dir)
            subprocess.run(ants_command_str, shell=True, check=True)

    def register_dti(self, subject):
        """
        Uses ANTs to warp and reorient DTI to template space
        """    

        converted_dir = self._conversion_folder(subject)
        registered_dti_dir = self._reg_folder(subject)
        template_folder = self.template_folder
        template_file = self.template_file

        if os.path.isdir(registered_dti_dir):
            registered_file = set(os.listdir(registered_dti_dir))
        else:
            registered_file = {}

        if registered_file == self.registered_dti_files:
            self.logger.info('registered dti files found for subject {}'.format(subject))
        else:
            self.logger.info('registering dti files for subject {}'.format(subject))
            if not os.path.isdir(registered_dti_dir):
                os.makedirs(registered_dti_dir)

            # 1) run ANTS registration
            ants_command_str = \
                'antsRegistrationSyN.sh -p f -f {0}/{3} -m {1}/FA.nii.gz -t s -o {2}/FA_reg_' \
                .format(template_folder, converted_dir, registered_dti_dir, template_file)
            subprocess.run(ants_command_str, shell=True, check=True)

            # 2) compose a single warp
            ants_command_str = \
                'antsApplyTransforms -d 3 -i {1}/FA.nii.gz -r {0}/{3} -t {2}/FA_reg_1Warp.nii.gz ' \
                '-t {2}/FA_reg_0GenericAffine.mat -o [ {2}/FA_reg_combinedWarp.nii.gz , 1 ]' \
                .format(template_folder, converted_dir, registered_dti_dir, template_file)
            subprocess.run(ants_command_str, shell=True, check=True)

            # 3) move DT to fixed
            ants_command_str = \
                'antsApplyTransforms -d 3 -e 2 -i {1}/DT.nii.gz -r {0}/{3} -t {2}/FA_reg_combinedWarp.nii.gz -o {2}/DTDeformed.nii.gz' \
                .format(template_folder, converted_dir, registered_dti_dir, template_file)
            subprocess.run(ants_command_str, shell=True, check=True)

            # 4) reorient warped DT
            ants_command_str = \
                'ReorientTensorImage 3 {0}/DTDeformed.nii.gz {0}/DTReorientedWarp.nii.gz {0}/FA_reg_combinedWarp.nii.gz' \
                .format(registered_dti_dir)
            subprocess.run(ants_command_str, shell=True, check=True)

    def get_eigen(self, subject):
        """
        Uses ANTs to get eigenvalues and eigenvectors from DTI
        """
        registered_dti_dir = self._reg_folder(subject)
        ants_dir = self._ants_folder(subject)

        if os.path.isdir(ants_dir):
            ants_file = set(os.listdir(ants_dir))
        else:
            ants_file = {}

        if ants_file == self.ants_dti_files:
            self.logger.info('re-processing dti files found for subject {}'.format(subject))
        else:
            self.logger.info('re-processing dti files for subject {}'.format(subject))
            if not os.path.isdir(ants_dir):
                os.makedirs(ants_dir)

            # get eigen values
            def return_eigenvector(indir, dt_image, vec_label, vec_idx, outdir, output):

                ants_command_str = \
                    'ImageMath 3 {3}/{4}.nii.gz TensorToVector {0}/{1} {2}' \
                    .format(indir, dt_image, vec_idx, outdir, vec_label)
                subprocess.run(ants_command_str, shell=True, check=True)

                for i in range(3):
                    ants_command_str = \
                        'ImageMath 3 {0}/{1}_{2}.nii.gz ExtractVectorComponent {0}/{1}.nii.gz {2}' \
                        .format(outdir, vec_label, i)
                    subprocess.run(ants_command_str, shell=True, check=True)

                ants_command_str = \
                    'ImageMath 4 {0}/{2} TimeSeriesAssemble 1 0 {0}/{1}_0.nii.gz {0}/{1}_1.nii.gz {0}/{1}_2.nii.gz' \
                    .format(outdir, vec_label, output)
                subprocess.run(ants_command_str, shell=True, check=True)

            return_eigenvector(registered_dti_dir, 'DTReorientedWarp.nii.gz', 'V1', 2, ants_dir, 'V1Deformed.nii.gz')
            return_eigenvector(registered_dti_dir, 'DTReorientedWarp.nii.gz', 'V2', 1, ants_dir, 'V2Deformed.nii.gz')
            return_eigenvector(registered_dti_dir, 'DTReorientedWarp.nii.gz', 'V3', 0, ants_dir, 'V3Deformed.nii.gz')
            # ------------

            # get eigen values
            ants_command_str = \
                'ImageMath 3 {1}/L1Deformed.nii.gz TensorEigenvalue {0}/DTReorientedWarp.nii.gz 2' \
                .format(registered_dti_dir, ants_dir)
            subprocess.run(ants_command_str, shell=True, check=True)

            ants_command_str = \
                'ImageMath 3 {1}/L2Deformed.nii.gz TensorEigenvalue {0}/DTReorientedWarp.nii.gz 1' \
                .format(registered_dti_dir, ants_dir)
            subprocess.run(ants_command_str, shell=True, check=True)

            ants_command_str = \
                'ImageMath 3 {1}/L3Deformed.nii.gz TensorEigenvalue {0}/DTReorientedWarp.nii.gz 0' \
                .format(registered_dti_dir, ants_dir)
            subprocess.run(ants_command_str, shell=True, check=True)

            # clean up
            file_list = glob.glob(os.path.join(ants_dir, '*_*.nii.gz'), recursive=False)
            for file in file_list: os.remove(file)
            os.remove(os.path.join(ants_dir, 'V1.nii.gz'))
            os.remove(os.path.join(ants_dir, 'V2.nii.gz'))
            os.remove(os.path.join(ants_dir, 'V3.nii.gz'))

    def build_dti_tensor_image(self, subject):
        """
        Reads in eigenvectors and eigenvalues from DTI fit and returns  3*3*i*j*k DTI array for input to nn
        """
        ants_dir = self._ants_folder(subject)

        dti_tensor = 0
        for i in range(1, 4):
            bvecs_file = glob.glob(os.path.join(ants_dir, 'V' + str(i) + '*'))[0]
            bvals_file = glob.glob(os.path.join(ants_dir, 'L' + str(i) + '*'))[0]
            bvecs = nib.load(bvecs_file).get_fdata()
            bvals = nib.load(bvals_file).get_fdata()
            dti_tensor = dti_tensor + np.einsum('abc, abci, abcj->ijabc', bvals, bvecs, bvecs)
        return dti_tensor
