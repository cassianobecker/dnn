import os
import torch.utils.data

from dataset.hcp.reader import HcpReader, SkipSubjectException
from util.logging import get_logger, set_logger
from fwk.config import Config


class HcpDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to host and dti diffusion data
    """

    def __init__(self, device, regime, coarsen=None):

        results_path = os.path.expanduser(Config.config['EXPERIMENT']['results_path'])

        if not os.path.exists(os.path.join(results_path, 'log')):
            os.mkdir(os.path.join(results_path, 'log'))
        log_furl = os.path.join(results_path, 'log', 'downloader.log')

        set_logger('HcpDataset', Config.config['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpDataset')
        self.logger.info('*** starting new {:} dataset'.format(regime))

        self.device = device
        self.reader = HcpReader()

        if Config.config.has_option('TRANSFORMS', 'region'):
            region_str = Config.config['TRANSFORMS']['region']
            self.region = self.reader.parse_region(region_str)
        else:
            self.region = None

        subject_file_url = Config.config['SUBJECTS'][f'{regime}_subjects_file']

        if 'max_subjects' in Config.config['SUBJECTS'].keys():
            max_subjects = Config.config['SUBJECTS']['max_subjects']
        else:
            max_subjects = None

        self.subjects = self.reader.load_subject_list(subject_file_url, max_subjects=max_subjects)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        return self.data_for_subject(subject, region=self.region)

    def data_for_subject(self, subject, region=None):

        try:
            self.reader.logger.info("feeding subject {:}".format(subject))

            dti_tensor = self.reader.load_dti_tensor_image(subject, region=region)
            target = self.reader.load_covariate(subject)

        except SkipSubjectException:
            self.reader.logger.warning("skipping subject {:}".format(subject))

        return dti_tensor, target, subject

    def tensor_size(self):
        tensor_shape = self.__getitem__(0)[0].shape
        return tensor_shape

    def self_check(self):
        for subject in self.subjects:
            self.reader.process_subject(subject)


class HcpDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super(HcpDataLoader, self).__init__(*args, **kwargs)


