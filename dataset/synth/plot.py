import numpy as np
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_streams(nodes=None, bundles=None):

    fig = plt.figure()
    ax = Axes3D(fig)

    num_points = 50

    if nodes is not None:

        for node in nodes:
            coords = list()
            for _ in range(num_points):
                coords.append(node.sample())

            coords = np.array(coords)
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker='o')

    if bundles is not None:

        for bundle in bundles:
            coords = np.array([stream for stream in bundle.streams])
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

            for cp in bundle.control_points:
                ax.scatter(cp.point[0], cp.point[1], cp.point[2], marker='s', s=40, c='black')

    plt.show()


def plot_track_vis(trk_url):
    url_exec = '/Applications/TrackVis.app/Contents/MacOS/track_vis'
    str_cmd = url_exec + ' ' + trk_url
    subprocess.run(str_cmd, shell=True, check=True)
