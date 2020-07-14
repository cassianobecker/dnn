import os
from os.path import join

import numpy as np
import numpy.random as npr

from dataset.synth.fibercup import create_fibercup
from dataset.synth.tract import Bundle, ControlPoint, Tractogram
from dataset.synth.plot import plot_track_vis


class FibercupRegressionDataset:

    def __init__(self):

        self.radius = 64
        self.depth = 6
        self.multiplier = 10

        tractogram, parcels = create_fibercup(radius=self.radius, depth=self.depth, mult=self.multiplier)

        self.tractogram: Tractogram = tractogram
        self.parcels = parcels

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

    def generate_tractogram_and_covariate(self):

        edge = (6, 7)

        self.tractogram.bundles.pop(edge, None)

        shift = npr.rand()
        offset = np.array([shift, 0, 0])
        bundle = self.create_bundle(edge, offset)

        self.tractogram.add(edge, bundle)

        return self.tractogram, shift

    def save_tract_and_label(self, path, tractogram, label, show_plot=False):

        os.makedirs(path, exist_ok=True)
        np.savetxt(join(path, 'label.txt'), np.array([label]), fmt='%f', delimiter='')

        fname = 'tracts'
        offset = [self.radius, self.radius, self.depth]
        tractogram.save(join(path, fname), offset)

        if show_plot is True:
            url_trk = join(path, fname + '.trk')
            plot_track_vis(url_trk)

        return join(path, fname + '.fib')

    def make_mask(self, path):
        mask_file_url = join(path, 'data_mask.nii.gz')
        self.parcels.save_mask(mask_file_url=mask_file_url)
