import os
import re
import random

from fwk.config import Config
from util.path import absolute_path
from util.lang import to_bool


class Subjects:

    @classmethod
    def create_list_from_config(cls):

        path = absolute_path(os.path.expanduser(Config.config['DATABASE']['local_processing_directory']))

        subjects = cls.list_from(path)

        if Config.config.has_option('SUBJECTS', 'shuffle'):
            if to_bool(Config.config['SUBJECTS']['shuffle']):
                random.shuffle(subjects)

        if Config.config.has_option('SUBJECTS', 'max_subjects'):
            max_subjects = int(Config.config['SUBJECTS']['max_subjects'])
            subjects = subjects[:max_subjects]

        if Config.config.has_option('SUBJECTS', 'percent_train'):
            percent_train = int(Config.config['SUBJECTS']['percent_train'])

            num_subjects = len(subjects)
            num_train = int(percent_train/100 * num_subjects)
            num_test = num_subjects - num_train

            train_subjects = subjects[:num_train]
            test_subjects = subjects[num_train:num_train + num_test]
        else:
            train_subjects = subjects
            test_subjects = []

        return train_subjects, test_subjects

    @staticmethod
    def list_from(path):
        abs_path = os.path.expanduser(os.path.join(path, 'HCP_1200_tensor'))
        files = sorted(os.listdir(abs_path))
        subject_pattern = '[0-9]{6}'
        subjects = [file for file in files if bool(re.match(subject_pattern, file))]
        return subjects

    @staticmethod
    def _partition(subjects, idxs):

        partitioned = []
        for idx in idxs:
            partitioned.append(subjects[:idx])
            del(subjects[:idx])

        return partitioned