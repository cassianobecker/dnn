import os
from os.path import join
import shutil
import subprocess
import numpy.random as npr
import numpy as np
import scipy.stats

from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, recursive_response
from dipy.reconst.csdeconv import auto_response
from dipy.core.gradients import gradient_table

from util.path import absolute_path
from dataset.synth.fibercup import create_fibercup
from dataset.synth.plot import plot_track_vis
from dataset.synth.tract import Bundle, ControlPoint, Tractogram

# general configurations
base_path = '~/mitk/dnn/.dnn/datasets'
dataset_name = 'synth4'

# docker paths for Fiberfox
docker_container_name = 'confident_nobel'
base_path_on_docker = '/dnn/.dnn/datasets'
fiberfox_executable = '/dnn/MitkDiffusion/MitkFiberfox.sh'


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def copy_param_dir(src_param_dir, dest_param_dir):
    if os.path.isdir(dest_param_dir):
        shutil.rmtree(dest_param_dir)
    shutil.copytree(src_param_dir, dest_param_dir)


class FibercupRegressionDataset:

    def __init__(self):

        self.base_path = os.path.expanduser(base_path)

        self.radius = 64
        self.depth = 6
        self.multiplier = 10

        tractogram, self.parcels = create_fibercup(radius=self.radius, depth=self.depth, mult=self.multiplier)
        self.tractogram: Tractogram = tractogram

        self.tracts_path = make_dir(join(self.base_path, dataset_name, 'tracts'))
        self.dwi_path = make_dir(join(self.base_path, dataset_name, 'dwi'))
        self.dti_path = make_dir(join(self.base_path, dataset_name, 'dti'))
        self.odf_path = make_dir(join(self.base_path, dataset_name, 'odf'))
        self.param_path = join(self.base_path, dataset_name, 'params')

        copy_param_dir(join(absolute_path('dataset'), 'synth', 'param'), join(self.base_path, dataset_name, 'params'))

        self.flip_evecs()

    def generate_samples(self, num_samples, show_plot=False):

        edge = (6, 7)

        for sample_id in range(num_samples):

            self.tractogram.bundles.pop(edge, None)

            shift = npr.rand()
            offset = np.array([shift, 0, 0])
            bundle = self.create_bundle(edge, offset)

            self.tractogram.add(edge, bundle)

            self.save_tract_and_label(sample_id, self.tractogram, label=shift, show_plot=show_plot)

            self.simulate_dwi(sample_id)

            self.fit_dti(sample_id)

            self.fit_odf(sample_id)

    def save_tract_and_label(self, sample_id, tractogram, label, show_plot=False):

        path = join(self.tracts_path, f'{sample_id}')
        if not os.path.isdir(path):
            os.makedirs(path)

        np.savetxt(join(path, 'label.txt'), np.array([label]), fmt='%f', delimiter='')

        fname = 'tracts'
        offset = [self.radius, self.radius, self.depth]
        tractogram.save(join(path, fname), offset)

        if show_plot is True:
            url_trk = join(path, fname + '.trk')
            plot_track_vis(url_trk)

        return join(path, fname + '.fib')

    def create_bundle(self, edge, offset):

        multiplier = 1
        ctl_pt_variance = 5
        weight = 200

        radius = self.radius
        depth = int(0.5 * self.depth + offset[2])

        control_points = [
            ControlPoint((-int(0.65 * radius + offset[0]), -int(0.3 * radius + offset[1]), depth), ctl_pt_variance),
            ControlPoint((-int(0.5 * radius + offset[0]), -int(0.4 * radius + offset[1]), depth), ctl_pt_variance),
            ControlPoint((-int(0.5 * radius + offset[0]), -int(0.5 * radius + offset[1]), depth), ctl_pt_variance),
            ControlPoint((-int(0.6 * radius + offset[0]), -int(0.7 * radius + offset[1]), depth), ctl_pt_variance)
        ]

        node0 = self.parcels.nodes[edge[0]]
        node1 = self.parcels.nodes[edge[1]]

        num_streams = weight * multiplier

        bundle = Bundle(node0, node1, control_points, num_streams)

        return bundle

    def simulate_dwi(self, sample_id):

        # make target directory on docker locally
        path = join(self.dwi_path, f'{sample_id}')
        if not os.path.isdir(path):
            os.makedirs(path)

        # define all paths relative to docker
        dwi_base_path = base_path_on_docker
        params_url = join(dwi_base_path, dataset_name, 'params', 'param.ffp')
        tracts_url = join(dwi_base_path, dataset_name, 'tracts', f'{sample_id}', 'tracts.fib')
        target_url = join(dwi_base_path, dataset_name, 'dwi', f'{sample_id}', 'data')

        docker_prefix = f'/usr/local/bin/docker exec -i {docker_container_name}'

        str_cmd = f'{docker_prefix} {fiberfox_executable} -o {target_url} -i {tracts_url} -p {params_url} --verbose'

        subprocess.run(str_cmd, shell=True, check=True)

    @staticmethod
    def _perform_dti_fit(dti_params, save_tensor=False):

        dti_fit_command_str = f"dtifit " \
                              f"-k {dti_params['data']} " \
                              f"-o {dti_params['output']} " \
                              f"-m {dti_params['mask']} " \
                              f"-r {dti_params['bvecs']} " \
                              f"-b {dti_params['bvals']} "

        if save_tensor is True:
            dti_fit_command_str += '--save_tensor'

        subprocess.run(dti_fit_command_str, shell=True, check=True)

    @staticmethod
    def get_otsu_mask(image):
        b0_mask, mask = median_otsu(image, median_radius=2, numpass=1, vol_idx=np.array([0, 1, 2]))
        return mask

    @staticmethod
    def get_mode_mask(image):

        masks = np.full(image.shape, 0)

        for k in range(3):
            mode_k = scipy.stats.mode(image[..., k].ravel())[0][0]
            masks[image[..., k] < 0.99 * mode_k] = 1

        mask = np.any(masks, axis=3) * 1.

        return mask

    def make_mask_from_dwi(self, sample_id, strategy='mode'):

        dwi_file_url = join(self.dwi_path, f'{sample_id}', 'data.nii.gz')
        image, affine = load_nifti(dwi_file_url)

        if strategy == 'mode':
            mask = self.get_mode_mask(image)

        elif strategy == 'otsu':
            mask = self.get_otsu_mask(image)

        else:
            raise ValueError('Not implemented: unknown dwi mask type')

        mask_file_url = join(self.dwi_path, f'{sample_id}', 'data_mask.nii.gz')
        affine = np.eye(4)
        save_nifti(mask_file_url, mask, affine)

    def make_mask(self, sample_id):
        mask_file_url = join(self.dwi_path, f'{sample_id}', 'data_mask.nii.gz')
        self.parcels.save_mask(mask_file_url=mask_file_url)

    def fit_dti(self, sample_id):

        output_dti_path = join(self.dti_path, f'{sample_id}')

        dti_params = {
            'data':  join(self.dwi_path, f'{sample_id}', 'data.nii.gz'),
            'mask': join(self.dwi_path, f'{sample_id}', 'data_mask.nii.gz'),
            'bvals': self._flipped_bvals_url(),
            'bvecs': self._flipped_bvecs_url(),
            'output': join(output_dti_path, 'dti')
        }

        if not os.path.isdir(output_dti_path):
            os.makedirs(output_dti_path)

        self.make_mask(sample_id)
        self._perform_dti_fit(dti_params, save_tensor=True)

        registered_tensor_url = join(self.dti_path, f'{sample_id}', 'dti_tensor.*')
        fslconvert_command_str = f'fslchfiletype NIFTI_GZ {registered_tensor_url}'
        subprocess.run(fslconvert_command_str, shell=True, check=True)

    def _flipped_bvals_url(self):
        return join(self.param_path, 'param_flipped.ffp.bvals')

    def _flipped_bvecs_url(self):
        return join(self.param_path, 'param_flipped.ffp.bvecs')

    def flip_evecs(self, flips=(1, -1, 1)):

        # flip eigenvectors for compatibility between Mitk Fiberfox and FSL dtifit
        bvals_url = join(self.param_path, 'param.ffp.bvals')
        bvecs_url = join(self.param_path, 'param.ffp.bvecs')
        bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)
        new_bvecs = bvecs @ np.diag(flips)
        return self.save_bvals_bvecs(bvals, new_bvecs)

    def save_bvals_bvecs(self, bvals, bvecs):
        np.savetxt(self._flipped_bvals_url(), np.expand_dims(bvals, axis=0), fmt='%d', delimiter='  ')
        np.savetxt(self._flipped_bvecs_url(), bvecs.T, fmt='%2.6f', delimiter='  ')

    def fit_odf(self, sample_id):

        # bvals_url = self._flipped_bvals_url()
        # bvecs_url = self._flipped_bvecs_url()
        bvals_url = join(self.param_path, 'param.ffp.bvals')
        bvecs_url = join(self.param_path, 'param.ffp.bvecs')

        bvals, bvecs = read_bvals_bvecs(bvals_url, bvecs_url)
        gtab = gradient_table(bvals, bvecs)

        volumes_url = join(self.dwi_path, f'{sample_id}', 'data.nii.gz')
        volumes, volumes_affine = load_nifti(volumes_url)

        response, ratio = auto_response(gtab, volumes, roi_center=(29, 48, 2), roi_radius=1, fa_thr=0.24)

        # response = recursive_response(gtab, volumes, sh_order=8,
        #                               peak_thr=0.01, init_fa=0.08,
        #                               init_trace=0.0021, iter=8, convergence=0.001,
        #                               parallel=True)

        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        csd_fit = csd_model.fit(volumes)
        odf = csd_fit.shm_coeff

        # mask_url = join(self.dwi_path, f'{sample_id}', 'data_mask.nii.gz')
        # self.make_mask(sample_id)
        # mask, affine = load_nifti(mask_url)
        # odf_masked = (odf.transpose((3, 0, 1, 2)) * mask).transpose((1, 2, 3, 0))
        odf_masked = odf

        output_odf_path = join(self.odf_path, f'{sample_id}')
        if not os.path.isdir(output_odf_path):
            os.makedirs(output_odf_path)

        odf_url = join(output_odf_path, 'odf.nii.gz')

        save_nifti(odf_url, odf_masked, volumes_affine)


if __name__ == '__main__':

    number_of_samples = 2
    dataset = FibercupRegressionDataset()
    dataset.generate_samples(number_of_samples, show_plot=False)
