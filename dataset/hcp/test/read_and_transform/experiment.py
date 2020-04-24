from dataset.hcp.reader import HcpReader
from util.path import append_path
import numpy as np
from util.plot import plot_tensor_slices


def _sparsity(tensor):
    sparsity = np.sum(tensor != 0) / np.sum(tensor != np.NaN)
    return sparsity


class ReadAndTransform:

    @staticmethod
    def execute():

        reader = HcpReader()

        max_subjects = 5
        subjects = reader.load_subject_list(append_path(__file__, 'conf/subjects.txt'), max_subjects=max_subjects)

        region = [(30, 120), (10, 50), (50, 100)]

        for subject in subjects:

            print(subject)
            img = reader.load_dti_tensor_image(subject, region=region)
            print(f'sparsity before {_sparsity(img)}')

            z = np.einsum('ijklm->klm', img ** 2)

            plot_tensor_slices(z, middle=True)

            # img2 = reader.apply_mask(img)

            # z2 = np.einsum('ijklm->klm', img2 ** 2)

            # plot_tensor_slices(z2, middle=True)

            # print(f'sparsity afeter {_sparsity(img2)}')

            pass
