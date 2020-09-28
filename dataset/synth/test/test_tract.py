import os
from dataset.synth.fibercup import create_fibercup
from dataset.synth.plot import plot_streams, plot_track_vis


def test_synth():

    radius = 64
    depth = 5
    mult = 1

    tractogram, parcels = create_fibercup(radius=radius, depth=depth, mult=mult)

    path = '/Users/cassiano/Dropbox/cob/work/upenn/research/projects/defri/code/fiberfox-wrapper/out'
    fname = 'fcup'
    offset = [radius, radius, depth]
    tractogram.save(os.path.join(path, fname), offset)

    url_trk = os.path.join(path, fname + '.trk')
    plot_track_vis(url_trk)

    pass


if __name__ == "__main__":
    test_synth()
