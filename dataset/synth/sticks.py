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


from dataset.synth.tract import Tractogram, ParcellatedCylinder, Bundle, ControlPoint


class StickDataset:

    def generate_dwi_and_covariate(self):
        shift = np.pi * npr.rand()
        vol = stick_phantom(shift)
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
                    datashape=(32, 32, 8, 65),
                    origin=(16, 16, 3),
                    scale=(12, 12, 10),
                    angles=np.linspace(0, 2 * np.pi, 32),
                    radii=np.linspace(0.2, 1, 6),
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


def stick_phantom(theta):

    fimg, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    bvecs[np.isnan(bvecs)] = 0
    gtab = gradient_table(bvals, bvecs)

    datashape = (32, 32, 8, 65)
    origin = (16, 16, 3)

    # sigma = (0.2, 0.2, 0)
    # uniform = 2 * npr.rand(3) - 1
    # shifted_origin = tuple([int((sigma[k] * uniform[k] + 1.) * origin[k]) for k in range(3)])

    shifted_origin = origin

    S0 = 100

    def f1(t):
        x = np.linspace(0, 0, len(t))
        y = np.linspace(-1, 1, len(t))
        z = np.zeros(x.shape)
        return x, y, z

    vol1 = orbital_phantom(gtab, datashape=datashape, origin=shifted_origin, func=f1, S0=S0)

    def f2(t):
        x = np.linspace(-1 * np.cos(theta), 1 * np.cos(theta), len(t))
        y = -np.linspace(-1 * np.sin(theta), 1 * np.sin(theta), len(t))
        z = np.zeros(x.shape)
        return x, y, z

    vol2 = orbital_phantom(gtab, datashape=datashape, origin=shifted_origin, func=f2, S0=S0)

    def f2(t):
        x = np.linspace(-1 * np.cos(theta), 1 * np.cos(theta), len(t))
        y = -np.linspace(-1 * np.sin(theta), 1 * np.sin(theta), len(t))
        z = np.zeros(x.shape)
        return x, y, z

    phi = np.pi * npr.rand()

    def f3(t):
        offset_y0 = 0.2
        offset_y1 = 0.6 * npr.rand() + offset_y0
        x = np.linspace(-1, 1, len(t))
        y = np.linspace(- offset_y1, - offset_y0, len(t))
        z = np.zeros(x.shape)
        return x, y, z

    vol3 = orbital_phantom(gtab, datashape=datashape, origin=shifted_origin, func=f3, S0=S0)

    vol = vol1 + vol2 + vol3

    return vol


def fibercup_phantom():

    S0 = 100

    fimg, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    bvecs[np.isnan(bvecs)] = 0
    gtab = gradient_table(bvals, bvecs)

    # datashape = (32, 32, 8, 65)
    # origin = (16, 16, 3)

    datashape = (64, 64, 8, 65)
    # origin = (-32, -32, -5)
    origin = (32, 32, 3)
    scale = (1, 1, 0.5)

    points_per_stream = 50
    t = np.linspace(-1, 1, points_per_stream)

    radius = 30
    depth = 1
    margin = 0.3
    mult = 1

    num_nodes = 12
    cp_var = 0.5

    parcels = ParcellatedCylinder(num_nodes, radius, depth, margin=margin)

    def traj1(t):
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

        x, y, z = np.hsplit(np.array(bundle.streams[0]), 3)
        return x[:, 0], y[:, 0], z[:, 0]

    def traj2(t):

        edge = (6, 7)
        weight = 1
        cps = [
            ControlPoint((-int(0.65 * radius), -int(0.3 * radius), int(0.5 * depth)), cp_var),
            # ControlPoint((-int(0.5 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var),
            # ControlPoint((-int(0.48 * radius), -int(0.45 * radius), int(0.5 * depth)), cp_var),
            ControlPoint((-int(0.5 * radius), -int(0.5 * radius), int(0.5 * depth)), cp_var),
            ControlPoint((-int(0.6 * radius), -int(0.7 * radius), int(0.5 * depth)), cp_var)
        ]
        node0 = parcels.nodes[edge[0]]
        node1 = parcels.nodes[edge[1]]
        num_streams = weight * mult
        bundle = Bundle(node0, node1, cps, num_streams)

        x, y, z = np.hsplit(np.array(bundle.streams[0]), 3)
        return x[:, 0], y[:, 0], z[:, 0]

    def traj3(t):
        edge = (8, 1)
        weight = 1
        cps = [
            ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
            ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)
        ]
        node0 = parcels.nodes[edge[0]]
        node1 = parcels.nodes[edge[1]]
        num_streams = weight * mult

        bundle = Bundle(node0, node1, cps, num_streams)

        x, y, z = np.hsplit(np.array(bundle.streams[0]), 3)
        return x[:, 0], y[:, 0], z[:, 0]

    vol1 = orbital_phantom(gtab, radii=np.linspace(0.2, 2, 6), datashape=datashape, origin=origin, scale=scale, func=traj1, t=t, S0=S0)
    vol2 = orbital_phantom(gtab, radii=np.linspace(0.2, 2, 6), datashape=datashape, origin=origin, scale=scale, func=traj2, t=t, S0=S0)
    vol3 = orbital_phantom(gtab, radii=np.linspace(0.2, 2, 6), datashape=datashape, origin=origin, scale=scale,
                           func=traj3, t=t, S0=S0)

    return vol1 + vol2 + vol3


if __name__ == '__main__':

    dwi = fibercup_phantom()

    path = os.path.expanduser(join('~/.dnn/datasets', 'phantom_fibercup', '0', 'dwi'))
    if not os.path.isdir(path):
        os.makedirs(path)

    save_nifti(join(path, 'data.nii.gz'), dwi, np.eye(4))

    pass
