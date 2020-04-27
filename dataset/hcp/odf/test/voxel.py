import numpy as np

from dipy.data import default_sphere, get_sphere
from dipy.reconst.shm import real_sym_sh_basis, sph_harm_ind_list, real_sph_harm
from dipy.viz import window, actor
from PIL import Image
from dipy.core.sphere import Sphere


def main():

    # sphere = default_sphere
    sphere = get_sphere('symmetric724')

    order = 4
    idxs = sph_harm_ind_list(sh_order=order)

    # shm_coeff = np.zeros(45)
    # shm_coeff[0] = 0
    # shm_coeff[1] = 1
    # shm_coeff[2] = 0
    #
    # sampling_matrix, m, n = real_sym_sh_basis(order, default_sphere.theta, default_sphere.phi)
    #
    # odf = np.dot(shm_coeff, sampling_matrix.T)

    for i in range(idxs[0].shape[0]):

        m = idxs[0][i]
        n = idxs[1][i]

        odf = real_sph_harm(m, n, sphere.theta, sphere.phi)

        for _ in range(3):
            odf = np.expand_dims(odf, axis=0)

        fodf_spheres = actor.odf_slicer(odf, sphere=sphere, scale=0.9, norm=False, colormap='plasma')
        # ren = window.Renderer()
        ren = window.Scene()
        ren.add(fodf_spheres)
        # window.record(ren, out_path='csd_peaks.png', size=(600, 600))
        pic = window.snapshot(ren, fname=None, size=(600, 600))
        Image.fromarray(pic).save('figs/csd_{:d}_{:d}.png'.format(m, n))
        window.show(ren)
        pass

    pass


if __name__ == '__main__':
    main()
