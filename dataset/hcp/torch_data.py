import os
import torch
import torch.utils.data
import configparser

from util.path import get_root
from util.logging import get_logger, set_logger

from dataset.hcp.hcp_data import HcpReader, SkipSubjectException
from dataset.hcp.transforms import SlidingWindow, TrivialCoarsening


def get_database_settings():
    """
    Creates a ConfigParser object with server/directory/credentials/logging info from preconfigured directory.
    :return: settings, a ConfigParser object
    """
    settings = configparser.ConfigParser()
    settings_furl = os.path.join(get_root(), 'dataset', 'hcp', 'conf', 'hcp_database.ini')
    settings.read(settings_furl)
    return settings


def empty_hcp_record():
    return None, None, None, None, None


class HcpDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to host and process the BOLD signal and the associated motor tasks
    """

    def __init__(self, params, device, regime, coarsen=None):

        database_settings = get_database_settings()

        log_furl = os.path.join(params['FILE']['experiment_path'], 'log', 'downloader.log')
        set_logger('HcpDataset', database_settings['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpDataset')
        self.logger.info('*** starting new {:} dataset'.format(regime))

        self.device = device
        self.session = params['SESSION'][regime]

        self.reader = HcpReader(database_settings, params)

        list_url = os.path.join(params['FILE']['experiment_path'], 'conf', regime, self.session, 'subjects.txt')
        self.subjects = self.reader.load_subject_list(list_url)

        if coarsen is None:
            coarsen = TrivialCoarsening()
        self.coarsen = coarsen

        self.transform = SlidingWindow(params['TIME_SERIES'], coarsen=coarsen)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        return self.data_for_subject(subject)

    def data_for_subject(self, subject):

        x_windowed, y_one_hot, graph_list_tensor, mapping_list_tensor, _ = empty_hcp_record()

        try:
            self.reader.logger.info("feeding subject {:}".format(subject))

            data = self.reader.process_subject(subject, [self.session])

            graph_list, mapping_list = self.coarsen(data['adjacency'])

            graph_list_tensor = self._to_tensor(graph_list, dtype=torch.long)
            mapping_list_tensor = self._to_tensor(mapping_list, dtype=torch.float)

            cues = data['functional'][self.session]['cues']
            ts = data['functional'][self.session]['ts']
            x_windowed, y_one_hot = self.transform(cues, ts, mapping_list)
            y_one_hot = torch.tensor(y_one_hot, dtype=torch.long).to(self.device)

        except SkipSubjectException:
            self.reader.logger.warning("skipping subject {:}".format(subject))

        return x_windowed, y_one_hot, graph_list_tensor, mapping_list_tensor, subject

    def data_shape(self):
        shape = self.reader.get_adjacency(self.subjects[0]).shape[0]
        return shape

    def self_check(self):
        for subject in self.subjects:
            self.reader.process_subject(subject, [self.session])

    def _to_tensor(self, graph_list, dtype):
        coos = [torch.tensor(graph, dtype=dtype).to(self.device) for graph in graph_list]
        return coos


class HcpDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        # batch size must be always 1 (per subject)
        batch_size = 1
        super(HcpDataLoader, self).__init__(*args, batch_size=batch_size, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        # filter empty items
        batch = list(filter(lambda x: x[0] is not None, batch))
        if len(batch) == 0:
            # if batch[0][0] is None:
            return empty_hcp_record()
        else:
            return torch.utils.data.dataloader.default_collate(batch)
