import os
import torch.utils.data

from dataset.mnist.reader import MnistReader, SkipImageException
from util.logging import get_logger, set_logger
from fwk.config import Config


class MnistDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to host and dti diffusion data
    """

    def __init__(self, device, images, regime, half_precision=False, max_img_channels=None):

        results_path = os.path.expanduser(Config.config['EXPERIMENT']['results_path'])

        if not os.path.exists(os.path.join(results_path, 'log')):
            os.mkdir(os.path.join(results_path, 'log'))
        log_furl = os.path.join(results_path, 'log', 'dataloader.log')

        set_logger('HcpDataset', Config.config['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpDataset')

        self.device = device
        self.half_precision = half_precision
        self.max_img_channels = max_img_channels

        self.scale = int(Config.get_option('TRANSFORMS', 'scale', 1))

        self.reader = MnistReader(regime)

        if Config.config.has_option('TRANSFORMS', 'region'):
            region_str = Config.config['TRANSFORMS']['region']
            self.region = self.reader.parse_region(region_str)
        else:
            self.region = None

        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return self.data_for_idx(
            image,
            region=self.region,
            max_img_channels=self.max_img_channels)

    def data_for_idx(self, idx, region=None, max_img_channels=None):

        dti_tensor, target = None, None

        try:
            self.reader.logger.info("feeding image {:}".format(idx))

            dti_tensor = self.reader.load_dwi_tensor_image(
                idx,
                region=region,
                max_img_channels=max_img_channels,
                scale=self.scale
            )

            target = self.reader.load_covariate(idx)

        except SkipImageException:
            self.reader.logger.warning("skipping images {:}".format(idx))

        return dti_tensor, target, idx

    def tensor_size(self):
        tensor_shape = self.__getitem__(0)[0].shape
        return tensor_shape


class MnistDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super(MnistDataLoader, self).__init__(*args, **kwargs)
