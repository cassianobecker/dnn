from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response
from dipy.sims.tests.test_phantom import test_phantom
from dipy.reconst.tests.test_csdeconv import test_recursive_response_calibration, test_csdeconv
from dipy.io.image import save_nifti
import numpy as np
# from dipy.sims.phantom import orbital_phantom
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.data import get_fnames
from dipy.reconst.shm import sf_to_sh

from dipy.sims.voxel import single_tensor, single_tensor_odf, diffusion_evals
from dipy.sims.phantom import diff2eigenvectors
from dipy.data import get_sphere, get_fnames, default_sphere, small_sphere
from dipy.core.geometry import sphere2cart
from scipy.spatial.transform import Rotation as R

from dipy.sims.voxel import _check_directions
from dipy.sims.voxel import all_tensor_evecs
from dipy.reconst.shm import sph_harm_ind_list

import time

from dipy.sims.voxel import cylinders_and_ball_soderman


from dipy.core.geometry import euler_matrix


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

    """Create a phantom based on a 3-D orbit ``f(t) -> (x,y,z)``.

    Parameters
    -----------
    gtab : GradientTable
        Gradient table of measurement directions.
    evals : array, shape (3,)
        Tensor eigenvalues.
    func : user defined function f(t)->(x,y,z)
        It could be desirable for ``-1=<x,y,z <=1``.
        If None creates a circular orbit.
    t : array, shape (K,)
        Represents time for the orbit. Default is
        ``np.linspace(0, 2 * np.pi, 1000)``.
    datashape : array, shape (X,Y,Z,W)
        Size of the output simulated data
    origin : tuple, shape (3,)
        Define the center for the volume
    scale : tuple, shape (3,)
        Scale the function before applying to the grid
    angles : array, shape (L,)
        Density angle points, always perpendicular to the first eigen vector
        Default np.linspace(0, 2 * np.pi, 32).
    radii : array, shape (M,)
        Thickness radii.  Default ``np.linspace(0.2, 2, 6)``.
        angles and radii define the total thickness options
    S0 : double, optional
        Maximum simulated signal. Default 100.
    snr : float, optional
        The signal to noise ratio set to apply Rician noise to the data.
        Default is to not add noise at all.

    Returns
    -------
    data : array, shape (datashape)

    See Also
    --------
    add_noise

    """

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

    # The entire volume is considered to be inside the brain.
    # Voxels without a fiber crossing through them are taken
    # to be isotropic with signal = S0.
    vol = np.zeros(datashape) + S0
    odf = np.zeros(list(datashape[:-1]) + [45])

    for i in range(len(dx)):
        evecs, R = diff2eigenvectors(dx[i], dy[i], dz[i])

        S = single_tensor(gtab, S0, evals, evecs)
        # S2 = single_tensor_odf(default_sphere.vertices, evals, evecs)
        # S2 = sf_to_sh(S2, default_sphere, 8, basis_type=basis_type[2])

        vol[int(x[i]), int(y[i]), int(z[i]), :] += S
        # odf[int(x[i]), int(y[i]), int(z[i]), :] += S2

        for r in radii:
            for j in range(len(angles)):
                rb = np.dot(R, np.array([bx[j], by[j], bz[j]]))

                ix = int(x[i] + r * rb[0])
                iy = int(y[i] + r * rb[1])
                iz = int(z[i] + r * rb[2])
                vol[ix, iy, iz] = vol[ix, iy, iz] + S
                # odf[ix, iy, iz] = odf[ix, iy, iz] + S2

    vol = vol / np.max(vol, axis=-1)[..., np.newaxis]
    vol *= S0

    # odf = odf / np.max(odf, axis=-1)[..., np.newaxis]
    # odf = np.where(odf == 0., odf, odf / np.linalg.norm(odf, axis=-1)[..., np.newaxis])

    # div = np.linalg.norm(odf, axis=-1)[..., np.newaxis]
    # odf = np.divide(odf, div, out=np.zeros_like(odf), where=div != 0)

    # if snr is not None:
    #     vol = add_noise(vol, snr, S0=S0, noise_type='rician')

    return vol, odf


def single_odf():
    basis_type = [None, 'tournier07', 'descoteaux07']

    N = 10
    sh_order = 8

    n_coeff = len(sph_harm_ind_list(sh_order)[0])

    shs = np.zeros((N, N, 1, n_coeff))

    # theta, phi = 45, 45

    thetas = np.linspace(0, 90, N)
    phis = np.linspace(0, 90, N)

    for i in range(len(thetas)):
        for j in range(len(phis)):

            theta = thetas[i]
            phi = phis[j]

            # evecs = all_tensor_evecs(sphere2cart(1, np.deg2rad(theta), np.deg2rad(phi)))

            evecs = all_tensor_evecs(_check_directions((theta, phi)))

            # evecs = R.from_euler('zyx', [90, 45, 30]).as_matrix()
            evals = diffusion_evals
            odf1 = single_tensor_odf(default_sphere.vertices, evals, evecs)

            theta = thetas[i]
            phi = -phis[j]

            evecs = all_tensor_evecs(sphere2cart(1, np.deg2rad(theta), np.deg2rad(phi)))

            odf2 = single_tensor_odf(default_sphere.vertices, evals, evecs)
            odf = 0.5 * odf1 + 0.5 * odf2

            sh = sf_to_sh(odf, default_sphere, sh_order, basis_type=basis_type[0])
            shs[i, j, 0, :] = sh

    # sh = np.expand_dims(sh, [0, 1, 2])
    save_nifti('odf_mosaic', shs, np.eye(4))

    pass


def save_voxel(url, data):
    data = np.expand_dims(data, [0, 1, 2])
    save_nifti(url, data, np.eye(4))


def phantom():
    fimg, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    bvecs[np.isnan(bvecs)] = 0
    gtab = gradient_table(bvals, bvecs)

    S0 = 100
    # N = 50

    # vol = orbital_phantom()

    theta = np.pi/18

    def f(t):
        x = np.sin(t)
        y = np.cos(t)
        # z=np.zeros(t.shape)
        z = np.linspace(-1, 1, len(x))
        return x, y, z

    # helix
    vol, odf = orbital_phantom(gtab, func=f, S0=S0)

    def f2(t):
        x = np.linspace(0, 0, len(t))
        y = np.linspace(-1, 1, len(t))
        z = np.zeros(x.shape)
        return x, y, z

    # first direction
    vol2, odf2 = orbital_phantom(gtab, func=f2, S0=S0)

    def f3(t):
        x = np.linspace(-1 * np.cos(theta), 1 * np.cos(theta), len(t))
        y = -np.linspace(-1 * np.sin(theta), 1 * np.sin(theta), len(t))
        z = np.zeros(x.shape)
        return x, y, z

    # second direction
    vol3, odf3 = orbital_phantom(gtab, func=f3, S0=S0)
    # double crossing
    vol23 = vol2 + vol3
    # odf23 = odf2 + odf3

    # """
    # def f4(t):
    #     x = np.zeros(t.shape)
    #     y = np.zeros(t.shape)
    #     z = np.linspace(-1, 1, len(t))
    #     return x, y, z
    #
    # # triple crossing
    # vol4, odf4 = orbital_phantom(gtab, func=f4, S0=S0)
    # vol234 = vol23 + vol4
    # odf234 = odf23 + odf4
    vol234 = vol23

    # save_nifti('odf', odf234, np.eye(4))

    # save_nifti('phantom', vol234, np.eye(4))
    #
    response = (np.array([0.0015, 0.0003, 0.0003]), S0)

    # response, ratio = auto_response(gtab, vol234, roi_radius=20, fa_thr=0.4)
    csd = ConstrainedSphericalDeconvModel(gtab, response)
    csd_fit = csd.fit(vol234)

    save_nifti('odf', csd_fit.shm_coeff, np.eye(4))


if __name__ == '__main__':
    # single_odf()

    start_time = time.time()
    phantom()
    print("--- %s seconds ---" % (time.time() - start_time))

    pass
