import numpy as np
import numpy.random as npr
import os
from os.path import join

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.sims.phantom import diff2eigenvectors
from dipy.sims.voxel import single_tensor, diffusion_evals


class PhantomRegressionDataset:

    def generate_dwi_and_covariate(self):
        shift = np.pi * npr.rand()
        vol = phantom(shift)
        return vol, shift

    def save_label(self, path, label):
        os.makedirs(path, exist_ok=True)
        np.savetxt(join(path, 'label.txt'), np.array([label]), fmt='%f', delimiter='')

    def save_dwi(self, path, dwi):
        save_nifti(join(path, 'data.nii.gz'), dwi, np.eye(4))


def orbital_phantom(gtab=None,
                    evals=diffusion_evals,
                    func=None,
                    t=np.linspace(0, 2 * np.pi, 1000),
                    datashape=(64, 64, 64, 65),
                    origin=(32, 32, 32),
                    scale=(25, 25, 25),
                    angles=np.linspace(0, 2 * np.pi, 32),
                    radii=np.linspace(0.2, 2, 6),
                    S0=100.,
                    snr=None):

    basis_type = [None, 'tournier07', 'descoteaux07']

    if gtab is None:
        fimg, fbvals, fbvecs = get_fnames('small_64D')
        gtab = gradient_table(fbvals, fbvecs)

    if func is None:
        x = np.sin(t)
        y = np.cos(t)
        z = np.zeros(t.shape)
    else:
        x, y, z = func(t)

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    x = scale[0] * x + origin[0]
    y = scale[1] * y + origin[1]
    z = scale[2] * z + origin[2]

    bx = np.zeros(len(angles))
    by = np.sin(angles)
    bz = np.cos(angles)

    vol = np.zeros(datashape) + S0

    for i in range(len(dx)):
        evecs, R = diff2eigenvectors(dx[i], dy[i], dz[i])

        S = single_tensor(gtab, S0, evals, evecs)

        vol[int(x[i]), int(y[i]), int(z[i]), :] += S

        for r in radii:
            for j in range(len(angles)):
                rb = np.dot(R, np.array([bx[j], by[j], bz[j]]))

                ix = int(x[i] + r * rb[0])
                iy = int(y[i] + r * rb[1])
                iz = int(z[i] + r * rb[2])
                vol[ix, iy, iz] = vol[ix, iy, iz] + S

    vol = vol / np.max(vol, axis=-1)[..., np.newaxis]
    vol *= S0

    return vol


def phantom(theta):

    fimg, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    bvecs[np.isnan(bvecs)] = 0
    gtab = gradient_table(bvals, bvecs)

    S0 = 100

    def f1(t):
        x = np.linspace(0, 0, len(t))
        y = np.linspace(-1, 1, len(t))
        z = np.zeros(x.shape)
        return x, y, z

    vol1 = orbital_phantom(gtab, func=f1, S0=S0)

    def f2(t):
        x = np.linspace(-1 * np.cos(theta), 1 * np.cos(theta), len(t))
        y = -np.linspace(-1 * np.sin(theta), 1 * np.sin(theta), len(t))
        z = np.zeros(x.shape)
        return x, y, z

    vol2 = orbital_phantom(gtab, func=f2, S0=S0)

    vol = vol1 + vol2

    return vol
