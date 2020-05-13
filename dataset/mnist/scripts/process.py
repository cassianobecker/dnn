from __future__ import print_function
from dataset.mnist.processor import MnistProcessor
from fwk.config import Config


class MnistProcessorScript:

    def execute(self):

        processor = MnistProcessor()

        image_batch_index = int(Config.config['IMAGES']['image_batch_index'])
        number_of_batches = int(Config.config['IMAGES']['number_of_batches'])
        dataset = Config.config['DATABASE']['mnist_dataset']

        if Config.config.has_option('IMAGES', 'max_images_per_batch'):
            max_img = int(Config.config['IMAGES']['max_images_per_batch'])
        else:
            max_img = None

        for regime in ['train', 'test']:

            print(f'\n**** Processing {regime} batch {image_batch_index + 1} of {number_of_batches} for -{dataset}-\n')

            image_idxs = processor.database.image_batch(image_batch_index, number_of_batches, regime)[:max_img]

            for image_idx in image_idxs:

                print(f'processing image {image_idx + 1} of {len(image_idxs)}')
                try:
                    # processor.process_image(image_idx, regime)
                    processor.process_labels(image_idx, regime)
                except Exception as e:
                    print(e)
