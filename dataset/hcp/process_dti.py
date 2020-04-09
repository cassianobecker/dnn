from __future__ import print_function

from dataset.hcp.torch_data import HcpDataset
from util.experiment import get_experiment_params


def main():

    params = get_experiment_params(__file__, __name__)

    process_set = HcpDataset(params, 'cpu', 'process')

    for subject in process_set.subjects:
        print('processing subject {}'.format(subject))
        try:
            process_set.reader.process_subject(subject, delete_folders=False)
            # process_set.reader.get_diffusion(subject)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
