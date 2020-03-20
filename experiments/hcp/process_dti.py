from __future__ import print_function
import argparse
import torch

from dataset.hcp.torch_data import HcpDataset
from util.experiment import get_experiment_params


def main():

    batch_size = 2
    test_batch_size = 10

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch HCP Diffusion Example')

    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    params = get_experiment_params(__file__, __name__)

    process_set = HcpDataset(params, device, 'process')

    for subject in process_set.subjects:
        print('processing subject {}'.format(subject))
        try:
            process_set.reader.process_subject(subject, delete_folders=True)
            # process_set.reader.get_diffusion(subject)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
