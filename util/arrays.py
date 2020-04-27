def slice_from_list_of_pairs(pair_list, null_offset=None):

    slice_list = []

    if null_offset is not None:
        for _ in range(null_offset):
            slice_list.append(slice(None))

    for pair in pair_list:
        slice_list.append(slice(pair[0], pair[1]))

    return tuple(slice_list)
