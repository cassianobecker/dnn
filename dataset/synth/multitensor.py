import os
from os.path import join

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from util.lang import class_for_name
from fwk.config import Config


class SynthProcessor:

    def __init__(self, dry_run=False):
        self.database_processing_path = os.path.expanduser(Config.get_option('DATABASE', 'local_processing_directory'))

    def make_path(self, sample_id, paths):
        path = join(self.database_processing_path, f'{sample_id}', paths)
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def process_subject(self, sample_id):
        self.simulate_files(sample_id)
        self.fit_odf(sample_id)

    def simulate_files(self, sample_id):

        phantom_class_name = Config.get_option('DWI', 'phantom_class_name')
        PhantomClass = class_for_name(phantom_class_name)
        dataset = PhantomClass()
        dwi, labels = dataset.generate_dwi_and_covariate()

        dwi_path = self.make_path(sample_id, 'dwi')
        dataset.save_dwi(dwi_path, dwi)

        label_path = self.make_path(sample_id, 'tracts')
        dataset.save_label(label_path, labels)

    def fit_odf(self, sample_id):

        fimg, fbvals, fbvecs = get_fnames('small_64D')
        bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
        bvecs[np.isnan(bvecs)] = 0
        gtab = gradient_table(bvals, bvecs)
        response = (np.array([0.0015, 0.0003, 0.0003]), 100)
        dwi_path = self.make_path(sample_id, 'dwi')
        dwi, affine = load_nifti(join(dwi_path, 'data.nii.gz'))

        csd = ConstrainedSphericalDeconvModel(gtab, response)
        csd_fit = csd.fit(dwi)

        odf_url = join(self.make_path(sample_id, 'odf'), 'odf.nii.gz')
        save_nifti(odf_url, csd_fit.shm_coeff, np.eye(4))
