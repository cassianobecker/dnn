import os
import torch
import torch.utils.data
import configparser

from util.path import get_root
from util.logging import get_logger, set_logger

from dataset.hcp.hcp_data import HcpReader, SkipSubjectException
# from dataset.hcp.transforms import SlidingWindow, TrivialCoarsening


def get_database_settings():
    """
    Creates a ConfigParser object with server/directory/credentials/logging info from preconfigured directory.
    :return: settings, a ConfigParser object
    """
    settings = configparser.ConfigParser()
    settings_furl = os.path.join(get_root(), 'dataset', 'hcp', 'conf', 'hcp_database.ini')
    settings.read(settings_furl)
    return settings


class HcpDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to host and process diffusion data
    """

    def __init__(self, params, device, regime, coarsen=None):

        database_settings = get_database_settings()

        if not os.path.exists(os.path.join(params['FILE']['experiment_path'], 'log')):
            os.mkdir(os.path.join(params['FILE']['experiment_path'], 'log'))
        log_furl = os.path.join(params['FILE']['experiment_path'], 'log', 'downloader.log')
        set_logger('HcpDataset', database_settings['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpDataset')
        self.logger.info('*** starting new {:} dataset'.format(regime))

        self.device = device

        self.reader = HcpReader(database_settings, params)

        list_url = os.path.join(params['FILE']['experiment_path'], 'conf', regime, 'subjects.txt')
        self.subjects = self.reader.load_subject_list(list_url)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        return self.data_for_subject(subject)

    def data_for_subject(self, subject):

        try:
            self.reader.logger.info("feeding subject {:}".format(subject))

            dti_tensor = self.reader.load_dti_tensor_image(subject)
            target = self.reader.load_covariate(subject)

        except SkipSubjectException:
            self.reader.logger.warning("skipping subject {:}".format(subject))

        return dti_tensor, target, subject

    def self_check(self):
        for subject in self.subjects:
            self.reader.process_subject(subject)


class HcpDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super(HcpDataLoader, self).__init__(*args, **kwargs)


