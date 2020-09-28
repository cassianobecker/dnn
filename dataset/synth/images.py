import os
import re
import random

from fwk.config import Config
from util.path import absolute_path
from util.lang import to_bool


class Images:

    @classmethod
    def create_list_from_config(cls):

        image_list = []
        for regime in ['train', 'test']:
            image_list.append(cls.list_for_regime(regime))

        return image_list

    @classmethod
    def list_for_regime(cls, regime):

        path = absolute_path(os.path.expanduser(Config.config['DATABASE']['local_processing_directory']))

        images = cls.list_from(path, regime)

        if Config.config.has_option('IMAGES', 'shuffle'):
            if to_bool(Config.config['IMAGES']['shuffle']):
                random.shuffle(images)

        if Config.config.has_option('IMAGES', f'max_{regime}_images'):
            max_images = int(Config.config['IMAGES'][f'max_{regime}_images'])

        return images[:max_images]

    @staticmethod
    def list_from(path, regime):
        abs_path = os.path.expanduser(os.path.join(path, regime))
        files = sorted(os.listdir(abs_path))
        subject_pattern = '[0-9]{1,5}'
        images = [file for file in files if bool(re.match(subject_pattern, file))]

        return images
