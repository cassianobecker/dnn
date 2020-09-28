from __future__ import print_function

from dataset.synth.dwi import SynthProcessor
from fwk.config import Config

from util.iterators import chunk_iterator
from util.lang import to_bool


class SynthProcessorScript:

    def execute(self):

        dry_run = Config.get_option('OUTPUTS', 'dry_run', cast_function=to_bool, default=False)

        processor = SynthProcessor(dry_run=dry_run)

        subjects, batch_index, number_of_batches = self.subject_iterator()

        print(f'\n**** Processing batch {batch_index + 1} of {number_of_batches}\n')

        for subject in subjects:
            print('processing subject {}'.format(subject))
            processor.process_subject(subject)

    def subject_iterator(self):

        number_of_subjects = int(Config.config['SUBJECTS']['number_of_subjects'])
        batch_index = int(Config.config['SUBJECTS']['subject_batch_index'])
        number_of_batches = int(Config.config['SUBJECTS']['number_of_batches'])
        max_sub = Config.get_option('SUBJECTS', 'max_subjects_per_batch', cast_function=int, default=None)

        subjects = chunk_iterator(batch_index, number_of_subjects, number_of_batches, max_sub)

        return subjects, batch_index, number_of_batches
