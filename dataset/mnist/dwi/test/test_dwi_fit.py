import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from os.path import expanduser, join

from dipy.viz import window, actor
from dipy.data import get_sphere
from dipy.reconst.dti import fractional_anisotropy, color_fa


def run_dwi_fit():

    digit_idx = 1
    digit_base_relative_path = '~/.dnn/datasets/hcp/processing_mnist/mnist/'
    digit_base_path = join(expanduser(digit_base_relative_path), f'{digit_idx}')

    digit_hardi_fname = join(digit_base_path, 'data.nii.gz')
    digit_hardi_bval_fname = join(digit_base_path, 'bvals')
    digit_hardi_bvec_fname = join(digit_base_path, 'bvecs')

    data_digit0, affine = load_nifti(digit_hardi_fname)
    digit_bvals, digit_bvecs = read_bvals_bvecs(digit_hardi_bval_fname, digit_hardi_bvec_fname)

    # digit_bvals = 1000. * (digit_bvals / digit_bvals)

    # digit_bvals[digit_bvals > 10] = 500

    digit_gtab = gradient_table(digit_bvals, digit_bvecs)

    data_digit = 100 * (np.abs(data_digit0))
    # data_digit = data_digit0

    tenmodel = dti.TensorModel(digit_gtab, fit_method='OLS')
    tenfit = tenmodel.fit(data_digit)

    #### DATA FOR HCP

    if False:

        template_subject = '100206'
        relative_path = '~/.dnn/datasets/hcp/mirror/HCP_1200'
        base_path = join(expanduser(relative_path), template_subject, 'T1w', 'Diffusion')

        hardi_fname = join(base_path, 'data.nii.gz')
        hardi_bval_fname = join(base_path, 'bvals')
        hardi_bvec_fname = join(base_path, 'bvecs')

        data0, affine = load_nifti(hardi_fname)
        bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
        gtab = gradient_table(bvals, bvecs)

        data = data0
        tenmodel = dti.TensorModel(gtab, fit_method='OLS')
        tenfit = tenmodel.fit(data)

    FA = fractional_anisotropy(tenfit.evals)

    FA[np.isnan(FA)] = 0
    # FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, tenfit.evecs)

    interactive = True

    sphere = get_sphere('repulsion724')

    ren = window.Renderer()

    # evals = tenfit.evals[13:43, 44:74, 28:29]
    # evecs = tenfit.evecs[13:43, 44:74, 28:29]

    evals = tenfit.evals[:, :, 1:2]
    evecs = tenfit.evecs[:, :, 1:2]

    # cfa = RGB[13:43, 44:74, 28:29]
    cfa = RGB
    cfa /= cfa.max()

    ren.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere, scale=0.3))

    print('Saving illustration as tensor_ellipsoids.png')

    window.record(ren, n_frames=1, out_path='tensor_ellipsoids.png', size=(600, 600))

    if interactive:
        window.show(ren)


if __name__ == '__main__':
    run_dwi_fit()
