from typing import overload

import numpy as np
import numpy.random as npr
from scipy import interpolate
import nibabel as nib

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import (create_nifti_header)
from dipy.tracking.distances import approx_polygon_track


class Tractogram:

    def __init__(self) -> None:
        super().__init__()
        self.bundles = dict()

    def add(self, edge, bundle):
        self.bundles[edge] = bundle

    def tolist(self, offset=None, downsample=True):

        all_streams = []
        for edge in self.bundles.keys():

            if offset is not None:
                streams = [stream + offset for stream in self.bundles[edge].streams]
            else:
                streams = self.bundles[edge].streams

            if downsample is True:
                streams = [approx_polygon_track(stream, 0.25) for stream in streams]

            all_streams.extend(streams)
        return all_streams

    def save(self, url, offset=None):
        header = create_nifti_header(np.eye(4), [100, 100, 10], [1., 1., 1.])
        sft = StatefulTractogram(self.tolist(offset), header, Space.RASMM)
        save_tractogram(sft, url + '.trk', bbox_valid_check=False)
        save_tractogram(sft, url + '.fib', bbox_valid_check=False)


class Node:

    def __init__(self) -> None:
        super().__init__()

    def sample(self):
        pass

    def points(self):
        pass


class Bundle:

    @overload
    def __getitem__(self, i: int):
        return self.streams[i]

    def __len__(self) -> int:
        return self.streams.__len__()

    def __init__(self, node0: Node, node1: Node,
                 control_points=None, num_streams=1, points_per_stream=50, kind=None, radius=1):
        super().__init__()

        self.points_per_stream = points_per_stream
        self.node0 = node0
        self.node1 = node1
        self.control_points = control_points if isinstance(control_points, list) else [control_points]
        self.streams = list()
        self._create(num_streams)
        self.kind = kind
        self.radius = radius

    def _create(self, num_streams):
        for _ in range(num_streams):
            stream = self.sample()
            self.streams.append(stream)

    def _interpolate_and_sample(self, s, t, v):

        f = None

        if self.kind is None:
            f = interpolate.interp1d(s, v, kind='cubic', bounds_error=False)

        return f(t)

    @staticmethod
    def _incremental_normalized_distances(points):
        s = np.array([0.] + [np.linalg.norm(points[k + 1] - points[k]) for k in range(points.shape[0] - 1)])
        return np.cumsum(s / np.sum(s))

    def sample(self):

        start = self.node0.sample()
        end = self.node1.sample()
        points = np.array([start, *[cp.sample() for cp in self.control_points], end])

        s = self._incremental_normalized_distances(points)
        t = np.linspace(0, 1, self.points_per_stream)

        # stream0 = [self._interpolate_and_sample(s, t, points[:, d]) for d in range(3)]

        cs = interpolate.CubicSpline(s, points)
        stream = cs(t)

        # return np.split(stream, 3, axis=1)

        return stream


class ControlPoint:

    def __init__(self, point, radius) -> None:
        super().__init__()
        self.point = point
        self.radius = radius

    def sample(self):
        return npr.multivariate_normal(self.point, self.radius * np.eye(3))


class ParcellatedCylinder:

    def __init__(self, num_nodes, radius, depth, margin=0) -> None:
        self.depth = depth
        self.radius = radius
        self.num_nodes = num_nodes
        self.margin = margin
        self.nodes = list()
        self._create()

    def _create(self):
        span = 2 * np.pi / self.num_nodes
        for node_idx in range(self.num_nodes):
            init_angle = node_idx * span
            node = CylindricalNode(init_angle, span, self.radius, self.depth, margin=self.margin)
            self.nodes.append(node)

    def save_mask(self, mask_file_url):

        mask = np.zeros((self.radius, self.radius, self.depth - 1))
        x, y, z = np.mgrid[: self.radius, :self.radius, :self.depth - 1]
        idx = (x - self.radius/2) ** 2 + (y - self.radius/2) ** 2 < (self.radius ** 2) / 4
        mask[idx] = 1

        # the affine values need to be set in accordance with fields
        # from params.ffp file for fiberfox
        # <spacing>
        #   <x> 2 </x>
        #   <y> 2 </y>
        #   <z> 2 </z>
        # </spacing>
        affine = np.diag([-2, -2, 2, 1])
        # <origin>
        #   <x> -2 * (self.radius - 1) </x> # = -126
        #   <y> -2 * (self.radius - 1) </y> # = -126
        #   <z> 3 </z>
        # </origin>
        affine[:3, 3] = [128, 128, 6]

        mask_img = nib.Nifti1Image(mask.astype(np.float32), affine)
        nib.save(mask_img, mask_file_url)

        pass


class CylindricalNode(Node):

    def __init__(self, init_angle, span, radius, depth, margin=0) -> None:
        super().__init__()
        self.init_angle = init_angle
        self.span = span
        self.radius = radius
        self.depth = depth
        self.margin = margin

    def sample(self):
        # angle = (1 + self.margin) * self.init_angle + self.span * npr.rand() / (1 + self.margin)
        angle = self.init_angle + self.span * (self.margin + (1 - 2 * self.margin) * npr.rand())
        depth = self.depth * npr.rand()
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        # z = depth
        z = 0

        return x, y, z
