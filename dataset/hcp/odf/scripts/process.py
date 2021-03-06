from __future__ import print_function

from dataset.hcp.odf.odf import HcpOdfProcessor
from fwk.config import Config
from util.path import append_path

import os
import numpy as np
from dipy.io.image import  save_nifti


class OdfProcessorScript:

    def execute(self):

        processor = HcpOdfProcessor()

        subject_batch_index = int(Config.config['SUBJECTS']['subject_batch_index'])
        number_of_batches = int(Config.config['SUBJECTS']['number_of_batches'])

        if Config.config.has_option('SUBJECTS', 'max_subjects_per_batch'):
            max_sub = int(Config.config['SUBJECTS']['max_subjects_per_batch'])
        else:
            max_sub = None

        print(f'\n**** Processing batch {subject_batch_index + 1} of {number_of_batches}\n')

        for subject in processor.database.subject_batch(subject_batch_index, number_of_batches)[:max_sub]:
            print('processing subject {}'.format(subject))
            try:
                # processor.process_subject(subject, delete_folders=False)
                transfer_odf(subject)
            except Exception as e:
                print(e)


def transfer_odf(subject):

    # ~/.dnn/datasets/hcp/processing_odf/HCP_1200_tensor/100206/tensor_100206.npz
    base_path = '~/.dnn/datasets/hcp/processing_odf/HCP_1200_tensor'
    base_url = os.path.expanduser(os.path.join(base_path, subject, f'tensor_{subject}.npz'))
    target_path = '~/.dnn/datasets/hcp/processing_dwi_rigid'
    target_url = os.path.expanduser(os.path.join(target_path, subject, 'moving', 'odf.nii.gz'))

    if os.path.isfile(base_url):
        data = np.load(base_url)['dwi_tensor']

        save_nifti(target_url, data, np.eye(4))

        if os.path.isfile(target_url):
            os.remove(base_url)


def test_batch_subjects():
    processor = HcpOdfProcessor()

    number_of_batches = int(Config.config['SUBJECTS']['number_of_batches'])

    print(Config.config['SUBJECTS']['subject_batch'])

    for b in range(number_of_batches):
        print(f'*** BATCH {b + 1} of {number_of_batches} ***')

        for (k, subject) in enumerate(processor.database.subject_batch(b, number_of_batches)):
            print(f'processing subject {k + 1}:  {subject}')


if __name__ == '__main__':
    config_url = append_path(__file__, 'conf/args.ini')
    Config.set_config_from_url(config_url)

    test_batch_subjects()
    OdfProcessorScript().execute()