from __future__ import print_function

from dataset.hcp.dwi.dwi import HcpDwiProcessor
from fwk.config import Config


class DwiProcessorScript:

    def execute(self):

        processor = HcpDwiProcessor()

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
                processor.process_subject(subject, delete_folders=False)
            except Exception as e:
                print(e)

