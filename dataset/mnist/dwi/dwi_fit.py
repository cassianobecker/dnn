import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from os.path import expanduser, join
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel


ground_truth = True
model = 'synth'

suffix = '_truth' if ground_truth is True else ''
base_relative_path = f'~/.dnn/datasets/{model}'
base_path = expanduser(base_relative_path)


def fit_dti():

    bvals, bvecs = load_bvals_bvecs()

    bvals = bvals[1:]
    bvecs = bvecs[1:, :]

    mov, affine = load_gradients_from_nifti()

    mov = mov[:, :, :, 1:]

    gtab = gradient_table(bvals, bvecs)

    tenmodel = dti.TensorModel(gtab, fit_method='OLS')
    tenfit = tenmodel.fit(mov)

    tensor_dti_url = join(base_path, 'tensor_dti' + suffix + '.nii.gz')
    save_nifti(tensor_dti_url, dti.lower_triangular(tenfit.quadratic_form), affine)


def fit_odf():

    bvals, bvecs = load_bvals_bvecs()
    mov, affine = load_gradients_from_nifti()
    gtab = gradient_table(bvals, bvecs)

    response, ratio = auto_response(gtab, mov, roi_radius=10, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)

    csd_fit = csd_model.fit(mov)
    tensor_odfs = csd_fit.shm_coeff

    tensor_odf_url = join(base_path, 'tensor_odf' + suffix + '.nii.gz')
    save_nifti(tensor_odf_url, tensor_odfs, affine)


def load_bvals_bvecs():
    digit_hardi_bval_url = join(base_path, 'bvals')
    digit_hardi_bvec_url = join(base_path, 'bvecs')
    bvals, bvecs = read_bvals_bvecs(digit_hardi_bval_url, digit_hardi_bvec_url)
    return bvals, bvecs


def load_gradients_from_numpy():

    digit_hardi_url = join(base_path,  model + '_mov' + suffix + '.npz')
    mov_dict = np.load(digit_hardi_url, allow_pickle=True)
    mov = mov_dict['mov']
    # mov = 100 * mov

    affine = np.eye(4)

    return mov, affine


def load_gradients_from_nifti():

    digit_hardi_url = join(base_path,  model + '_mov' + suffix + '.nii.gz')
    mov, affine = load_nifti(digit_hardi_url)
    # mov = 100 * mov

    return mov, affine


if __name__ == '__main__':
    fit_dti()
    fit_odf()
