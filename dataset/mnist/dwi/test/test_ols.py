import numpy as np

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import matplotlib.pyplot as plt
import dipy.reconst.dti as dti
import os


def fit_simple():

    from dipy.data import get_fnames

    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    rel_path = '/Users/cassiano/.dnn/datasets/stanford'

    # hardi_fname = rel_path + 'data.nii.gz'
    # hardi_bval_fname = rel_path + 'bvals'
    # hardi_bvec_fname = rel_path + 'bvecs'

    data0, affine = load_nifti(hardi_fname)

    mask = np.zeros_like(data0)

    mask[data0 > 1.e-6] = 1

    save_mask(mask, affine, rel_path)


    data = data0[13: 43, 44: 74, 28: 29, :]

    # data = data0[13: 14, 44: 45, 28: 29, :]

    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    tensor_model = dti.TensorModel(gtab, fit_method='OLS', return_S0_hat=True)

    tensor_fit = tensor_model.fit(data)
    dti_coeffs = dti.lower_triangular(tensor_fit.quadratic_form)

    dti_coeffs2, S0_hat = ols(bvals, bvecs, data)

    i = 5
    j = 5

    for k in range(10,100):
        D = np.squeeze(tensor_fit.quadratic_form[i, j, :, :, :])
        g = np.array(bvecs.tolist()[k])
        ratioS = np.exp(-bvals[k] * g.T @ D @ g)
        S = data[i, j, :, k]
        S0 = tensor_fit.S0_hat[i, j, :]
        print(f'SoverS0= {S/S0}, ratioS = {ratioS}')


    # close = np.allclose(dti_coeffs, dti2_coeffs)
    # print(f'All close: {close}')

    return dti_coeffs, data, affine


def ols(bvals, bvecs, data):

    G = design_matrix(bvals, bvecs)

    Ginv = np.linalg.pinv(G)

    result = np.einsum('ij,...j->...i', Ginv, np.log(data))

    dti_coeffs = result[:, :, :, :-1]
    S0_hat = np.exp(result[:, :, :, -1])

    return dti_coeffs, S0_hat


def design_matrix(bvals, bvecs):
    # B = np.zeros((gtab.gradients.shape[0], 7))
    # B[:, 0] = gtab.bvecs[:, 0] * gtab.bvecs[:, 0] * 1. * gtab.bvals   # Bxx
    # B[:, 1] = gtab.bvecs[:, 0] * gtab.bvecs[:, 1] * 2. * gtab.bvals   # Bxy
    # B[:, 2] = gtab.bvecs[:, 1] * gtab.bvecs[:, 1] * 1. * gtab.bvals   # Byy
    # B[:, 3] = gtab.bvecs[:, 0] * gtab.bvecs[:, 2] * 2. * gtab.bvals   # Bxz
    # B[:, 4] = gtab.bvecs[:, 1] * gtab.bvecs[:, 2] * 2. * gtab.bvals   # Byz
    # B[:, 5] = gtab.bvecs[:, 2] * gtab.bvecs[:, 2] * 1. * gtab.bvals   # Bzz
    # B[:, 6] = np.ones(gtab.gradients.shape[0])
    
    n_grads = len(bvals)
    
    B = np.zeros((n_grads, 7))
    B[:, 0] = bvecs[:, 0] * bvecs[:, 0] * 1. * bvals   # Bxx
    B[:, 1] = bvecs[:, 0] * bvecs[:, 1] * 2. * bvals   # Bxy
    B[:, 2] = bvecs[:, 1] * bvecs[:, 1] * 1. * bvals   # Byy
    B[:, 3] = bvecs[:, 0] * bvecs[:, 2] * 2. * bvals   # Bxz
    B[:, 4] = bvecs[:, 1] * bvecs[:, 2] * 2. * bvals   # Byz
    B[:, 5] = bvecs[:, 2] * bvecs[:, 2] * 1. * bvals   # Bzz
    B[:, 6] = np.ones(n_grads)
    
    return -B


def save_dti(dti_coeffs, data, affine, relative_path):

    name = 'dti.nii.gz'

    path = os.path.expanduser(relative_path)
    url = os.path.join(path, name)

    if not os.path.isdir(path):
        os.makedirs(path)

    # affine = np.eye(4)
    dti_coeffs = dti_coeffs[:, :, :, (0, 1, 3, 2, 4, 5)]
    save_nifti(url, dti_coeffs, affine)


def save_mask(mask, affine, relative_path):

    name = 'mask.nii.gz'

    path = os.path.expanduser(relative_path)
    url = os.path.join(path, name)

    if not os.path.isdir(path):
        os.makedirs(path)

    # affine = np.eye(4)
    save_nifti(url, mask, affine)



def test_ols():
    relative_path = '~/.dnn/datasets/ols'
    dti_coeffs, data, affine = fit_simple()
    save_dti(dti_coeffs, data, affine, relative_path)


if __name__ == "__main__":
    test_ols()
