from __future__ import print_function

from dataset.hcp.odf.odf import HcpOdfProcessor
from fwk.config import Config
from util.path import append_path


def main():

    processor = HcpOdfProcessor()

    max_subjects = 2

    for subject in processor.database.subjects[:max_subjects]:

        print('processing subject {}'.format(subject))
        try:
            processor.process_subject(subject, delete_folders=False)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    config_url = append_path(__file__, 'conf/args.ini')
    Config.set_config_from_url(config_url)
    main()
