from __future__ import print_function

from dataset.hcp.torch_data import HcpDataset
from fwk.config import Config
from util.path import append_path


def main():

    process_set = HcpDataset('cpu', 'process')

    for subject in process_set.subjects:
        print('processing subject {}'.format(subject))
        try:
            process_set.reader.process_subject(subject, delete_folders=False)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    config_url = append_path(__file__, 'conf/args.ini')
    Config.set_config_from_url(config_url)
    main()
