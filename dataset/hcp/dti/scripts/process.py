from __future__ import print_function

from dataset.hcp.dti.dwi import HcpProcessor
from fwk.config import Config
from util.path import append_path


def main():

    processor = HcpProcessor()

    for subject in processor.subjects:
        print('processing subject {}'.format(subject))
        try:
            processor.process_subject(subject, delete_folders=False)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    config_url = append_path(__file__, 'conf/args.ini')
    Config.set_config_from_url(config_url)
    main()
