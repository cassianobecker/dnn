import numpy as np
import nibabel as nib
from scipy.stats import mode


def make_mask(image):

    def get_mode(array):
        return mode(array.ravel())[0][0]

    masks = np.full(image.shape, False)
    for k in range(3, image.shape[-1]):
        masks[image[..., k] == get_mode(image[..., k])] = 0
        masks[image[..., k] > 55] = True
    return np.any(masks, axis=3) * 1.


def create_nifti_header(dimensions=(30, 30, 20), voxel_sizes=(2., 2., 2.)):
    #     affine = np.array(
    #         [[   2.,    0.,    0.,  -80.],
    #          [   0.,    2.,    0., -120.],
    #          [   0.,    0.,    2.,  -60.],
    #          [   0.,    0.,    0.,    1.]]
    #     )

    affine = np.array(
        [[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]]
    )

    header = nib.nifti1.Nifti1Header()
    header['srow_x'] = affine[0, 0:4]
    header['srow_y'] = affine[1, 0:4]
    header['srow_z'] = affine[2, 0:4]
    header['dim'] = np.array([0] + dimensions + [1] * 4, dtype=np.int16)
    header['pixdim'] = np.array([1.] + voxel_sizes + [1.] * 4, dtype=np.float32)

    return header
