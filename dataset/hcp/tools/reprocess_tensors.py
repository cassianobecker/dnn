import os
import numpy as np
import re


def process_odf_tensor(abs_path, subject):

    print(f'loading odf tensor for subject {subject}:')

    tensor_name = 'odf_tensor_' + subject + '.npz'
    tensor_url = os.path.join(abs_path, subject, tensor_name)

    if not os.path.isfile(tensor_url):
        print(f'tensor file not found: {tensor_url}')
        return

    tensor_dict = np.load(tensor_url)
    tensor = next(iter(tensor_dict.values()))

    print(f'reshaping odf tensor for subject {subject}:')
    # check if dimensions need to be permuted (for odf pre-processing)
    if tensor.shape[3] == 45:
        tensor_reshaped = np.transpose(tensor, (3, 0, 1, 2))
    else:
        tensor_reshaped = tensor

    fname = 'tensor_' + subject
    new_tensor_url = os.path.join(abs_path, subject, fname)

    print(f'saving odf tensor for subject {subject}:')
    np.savez_compressed(new_tensor_url, dwi_tensor=tensor_reshaped)

    print(f'deleting odf tensor for subject {subject}:')
    os.remove(tensor_url)

    print(f'subject {subject} done.\n')


def process_dti_tensor(abs_path, subject):

    print(f'loading dti tensor for subject {subject}:')

    tensor_name = 'dti_tensor_' + subject + '.npz'
    tensor_url = os.path.join(abs_path, subject, tensor_name)

    if not os.path.isfile(tensor_url):
        print(f'tensor file not found: {tensor_url}')
        return

    tensor_dict = np.load(tensor_url)
    tensor = next(iter(tensor_dict.values()))

    fname = 'tensor_' + subject
    new_tensor_url = os.path.join(abs_path, subject, fname)

    print(f'saving dti tensor for subject {subject}:')
    np.savez_compressed(new_tensor_url, dwi_tensor=tensor)

    print(f'deleting dti tensor for subject {subject}:')
    os.remove(tensor_url)

    print(f'subject {subject} done.\n')


def process_all_odf_tensors(path):

    abs_path = os.path.expanduser(path)
    subjects = sorted(os.listdir(abs_path))

    for i, subject in enumerate(subjects):
        if bool(re.match('\d{6}', subject)):
            print(f'---- {i} of {len(subjects)} ---')
            process_odf_tensor(abs_path, subject)


def process_all_dti_tensors(path):

    abs_path = os.path.expanduser(path)
    subjects = sorted(os.listdir(abs_path))

    for i, subject in enumerate(subjects):
        if bool(re.match('\d{6}', subject)):
            print(f'---- {i} of {len(subjects)} ---')
            process_dti_tensor(abs_path, subject)


if __name__ == '__main__':

    # path = '~/.dnn/datasets/hcp/processing_odf/HCP_1200_tensor'
    # process_all_odf_tensors(path)

    path = '~/.dnn/datasets/hcp/processing/HCP_1200_tensor'
    process_all_dti_tensors(path)
