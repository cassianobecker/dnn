import os
import torch.utils.data

from dataset.hcp.reader import HcpReader, SkipSubjectException
from util.logging import get_logger, set_logger
from util.lang import to_bool
from fwk.config import Config
import numpy.random as npr
import numpy as np


class HcpDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to host and dti diffusion data
    """

    def __init__(self, device, subjects, half_precision=False, max_img_channels=None):

        results_path = os.path.expanduser(Config.config['EXPERIMENT']['results_path'])

        if not os.path.exists(os.path.join(results_path, 'log')):
            os.mkdir(os.path.join(results_path, 'log'))
        log_furl = os.path.join(results_path, 'log', 'dataloader.log')

        set_logger('HcpDataset', Config.config['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpDataset')

        self.device = device
        self.half_precision = half_precision
        self.max_img_channels = max_img_channels

        self.perturb = to_bool(Config.get_option('DATABASE', 'perturb', 'False'))

        self.reader = HcpReader()

        if Config.config.has_option('TRANSFORMS', 'region'):
            region_str = Config.config['TRANSFORMS']['region']
            self.region = self.reader.parse_region(region_str)
        else:
            self.region = None

        self.scale = int(Config.get_option('TRANSFORMS', 'scale', 1))

        self.subjects = subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        return self.data_for_subject(
            subject,
            region=self.region,
            max_img_channels=self.max_img_channels,
            perturb=self.perturb
        )

    def data_for_subject(self, subject, region=None, max_img_channels=None, perturb=False):

        dti_tensor, target = None, None

        try:
            self.reader.logger.info("feeding subject {:}".format(subject))

            dwi_tensor = self.reader.load_dwi_tensor_image(
                subject,
                region=region,
                max_img_channels=max_img_channels,
                scale=self.scale,
                perturb=perturb
            )

            target = self.reader.load_covariate(subject)

            if to_bool(Config.get_option('DATABASE', 'randomize', 'False')):
                dwi_tensor = self._randomize_dwi_tensor(dwi_tensor, target)

        except SkipSubjectException:
            self.reader.logger.warning("skipping subject {:}".format(subject))

        return dwi_tensor, target, subject

    def _randomize_dwi_tensor(self, dwi_tensor, target):
        if target.argmax() == 1:
            dwi_tensor = dwi_tensor * (2 * npr.rand(*dwi_tensor.shape) - 1)
        return dwi_tensor.astype(np.double)

    def tensor_size(self):
        tensor_shape = self.__getitem__(0)[0].shape
        return tensor_shape

    def number_of_classes(self):
        num_classes = self.__getitem__(0)[1].size
        return num_classes


class HcpDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super(HcpDataLoader, self).__init__(*args, **kwargs)
