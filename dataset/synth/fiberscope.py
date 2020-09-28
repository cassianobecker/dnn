from dataset.synth.tract import Tractogram, ParcellatedCylinder, Bundle, ControlPoint
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
from dipy.sims.voxel import multi_tensor

class FiberScopeDataset:

    @staticmethod
    def generate_dwi_and_covariate():

        shift = npr.rand(4)
        perturb = Perturb()
        perturb['covar_0'] = shift[0]
        perturb['covar_1'] = shift[1]
        perturb['covar_2'] = shift[2]
        perturb['covar_3'] = shift[3]

        vol = fiberscope_phantom(perturb)
        return vol, perturb

    def save_label(self, path, perturb):
        os.makedirs(path, exist_ok=True)
        for key in perturb:
            np.savetxt(join(path, key + '.txt'), np.array([perturb[key]]), fmt='%f', delimiter='')

    def save_dwi(self, path, dwi):
        save_nifti(join(path, 'data.nii.gz'), dwi, np.eye(4))


def phantom(bundle,
            gtab=None,
            evals=diffusion_evals,
            datashape=(32, 32, 8, 65),
            origin=(16, 16, 3),
            scale=(12, 12, 10),
            angles=np.linspace(0, 2 * np.pi, 32),
            radii=np.linspace(0.2, 1, 6),
            S0=100.,
            snr=None):

    # basis_type = [None, 'tournier07', 'descoteaux07']

    if gtab is None:
        fimg, fbvals, fbvecs = get_fnames('small_64D')
        gtab = gradient_table(fbvals, fbvecs)

    x, y, z = [component[:, 0] for component in np.hsplit(np.array(bundle.streams[0]), 3)]
    # return x[:, 0], y[:, 0], z[:, 0]

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


class Perturb(dict):
    """
    simple dictionary function that returns 0 if key does not exist
    """

    def __getitem__(self, k):
        return super().get(k, 0)


def fiberscope_phantom(perturb=None):

    S0 = 100

    fimg, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    bvecs[np.isnan(bvecs)] = 0
    gtab = gradient_table(bvals, bvecs)

    radius = 40
    depth = 10
    guard = 4
    parcellation_margin = 0.3
    points_per_stream = 200

    tractogram = get_fiberscope(
        perturb=perturb,
        radius=radius,
        depth=depth,
        margin=parcellation_margin,
        points_per_stream=points_per_stream)

    data_shape = (2 * (radius + guard), 2 * (radius + guard), depth, bvals.shape[0])
    origin = (radius + guard - 1, radius + guard - 1, 0)
    scale = (1, 1, 0.5)

    vol = 0
    for bundle in list(tractogram.bundles.values()):
        radii = np.linspace(0.2, 2 * bundle.radius, 10)
        vol += phantom(
            bundle,
            gtab,
            radii=radii,
            datashape=data_shape,
            origin=origin,
            scale=scale,
            S0=S0
        )

    return vol


def get_fiberscope(perturb=None, radius=64, depth=5, margin=0.3, mult=500, points_per_stream=50):

    if perturb is None:
        perturb = Perturb()

    num_nodes = 12
    cp_var = 1

    parcels = ParcellatedCylinder(num_nodes, radius, depth, margin=margin)

    tractogram = Tractogram()

    edge = (3, 2)
    weight = 1
    cps = [
        ControlPoint((-int(0.32 * radius), int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.25 * radius), int(0.4 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.0 * radius), int(0.3 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.25 * radius), int(0.4 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.32 * radius), int(0.6 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams, points_per_stream=points_per_stream)

    tractogram.add(edge, bundle)

    # ##########

    edge = (4, 9)
    weight = 1
    cps = [
        ControlPoint((-int(0.5 * radius), -int(0. * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.25 * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var)
    ]

    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams, points_per_stream=points_per_stream)

    tractogram.add(edge, bundle)

    # ##########

    edge = (6, 7)
    weight = 1
    cps = [
        ControlPoint((-int(0.65 * radius), -int(0.3 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.5 * radius + perturb['covar_3']), -int(0.4 * radius), int(0.5 * depth)), cp_var),
        # ControlPoint((-int(0.48 * radius), -int(0.45 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.5 * radius + perturb['covar_3']), -int(0.5 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.6 * radius), -int(0.7 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams, points_per_stream=points_per_stream)

    tractogram.add(edge, bundle)

    # ##########

    edge = (5, 0)
    weight = 1
    cps = []
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams, points_per_stream=points_per_stream)

    bundle.radius = 0.3 + 2 * perturb['covar_0']

    tractogram.add(edge, bundle)

    # ##########

    edge = (8, 1)
    weight = 1
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams, points_per_stream=points_per_stream)

    tractogram.add(edge, bundle)

    # ##########

    edge = (8, 0)
    weight = 1
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams, points_per_stream=points_per_stream)

    tractogram.add(edge, bundle)

    # ##########

    edge = (8, 11)
    weight = 1
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius) + perturb['covar_2'], int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams, points_per_stream=points_per_stream)

    tractogram.add(edge, bundle)

    return tractogram


if __name__ == '__main__':

    dwi = fiberscope_phantom()

    path = os.path.expanduser(join('~/.dnn/datasets', 'phantom_fibercup', '0', 'dwi'))
    if not os.path.isdir(path):
        os.makedirs(path)

    save_nifti(join(path, 'data.nii.gz'), dwi, np.eye(4))

    pass
