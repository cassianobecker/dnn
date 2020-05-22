import numpy as np

from dipy.align.metrics import CCMetric
from dipy.align.imaffine import transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D


def find_rigid_affine(static, static_affine,  moving, moving_affine):

    static_grid2world = static_affine
    moving_grid2world = moving_affine

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    return rigid


def find_prealign_affine(static, static_affine,  moving, moving_affine):

    static_grid2world = static_affine
    moving_grid2world = moving_affine

    # identity = np.eye(4)
    # affine_map = AffineMap(identity,
    #                        static.shape, static_grid2world,
    #                        moving.shape, moving_grid2world)

    # resampled = affine_map.transform(moving)

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)

    # transformed = c_of_mass.transform(moving)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    # transformed = translation.transform(moving)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    # transformed = rigid.transform(moving)

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    return affine.affine


def find_warp(pre_align_affine, static, static_affine, moving, moving_affine):

    # affine_map = AffineMap(pre_align_affine, static.shape, static_affine,moving.shape, moving_affine)
    # resampled = affine_map.transform(moving)

    metric = CCMetric(3)
    level_iters = [10, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    mapping = sdr.optimize(static, moving, static_affine, moving_affine, pre_align_affine)

    return mapping


def apply_warp_on_volumes(data, mapping):
    warped_data = np.zeros_like(data)
    for k in range(data.shape[-1]):
        print(f'embedding {k}')
        warped_data[..., k] = mapping.transform(data[..., k])

    return warped_data
