import os
import numpy as np
from util.path import get_root


def convert_surf_to_nifti(faces, coords, hemi):
    rows, cols = get_row_cols(faces, hemi)
    new_coords = filter_surf_vertices(coords)
    return rows, cols, new_coords


def read_surf_to_gray_map(hemi):
    fname = os.path.join(get_root(), 'dataset', 'hcp', 'res', hemi + '_dense_map.txt')
    surf_to_gray = np.loadtxt(fname, delimiter=',', dtype=int)
    return surf_to_gray


def map_to_surf(idx, surf_to_gray):
    surf_idx = np.nonzero(surf_to_gray[:, 1] == idx)[0]

    if surf_idx.shape[0] == 0:
        to_idx = -1
    else:
        to_idx = surf_to_gray[int(surf_idx), 0]

    return to_idx


def get_row_cols(faces, hemi):
    rows = list()
    cols = list()

    surf_to_gray = read_surf_to_gray_map(hemi)

    for i in faces:

        p1 = map_to_surf(i[0], surf_to_gray)
        p2 = map_to_surf(i[1], surf_to_gray)
        p3 = map_to_surf(i[2], surf_to_gray)

        if p1 > 0 and p2 > 0:
            rows.append(p1)
            cols.append(p2)
            rows.append(p2)
            cols.append(p1)

        if p1 > 0 and p3 > 0:
            rows.append(p1)
            cols.append(p3)
            rows.append(p3)
            cols.append(p1)

        if p2 > 0 and p3 > 0:
            rows.append(p2)
            cols.append(p3)
            rows.append(p3)
            cols.append(p2)

    return rows, cols


def filter_surf_vertices(coords):
    new_coords = []

    hemi = 'L'
    surf_to_gray = read_surf_to_gray_map(hemi)
    for i in range(surf_to_gray.shape[0]):
        idx_old = surf_to_gray[i, 1]
        idx_new = surf_to_gray[i, 0]
        new_coords.insert(idx_new, coords[idx_old, :])

    hemi = 'R'
    surf_to_gray = read_surf_to_gray_map(hemi)
    for i in range(surf_to_gray.shape[0]):
        idx_old = surf_to_gray[i, 1]
        idx_new = surf_to_gray[i, 0]
        new_coords.insert(idx_new, coords[idx_old, :])

    new_coords = np.array(new_coords)

    return new_coords
