from fwk.metrics import Metric
from fwk.config import Config


class EpochCounter(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.epoch = None
        self.total_epochs =  None

    def extract_metric(self, local_variables: dict):

        self.epoch = self.get_variable(local_variables, 'epoch')
        self.total_epochs = Config.config['ALGORITHM']['epochs']

    def print_metric(self):
        print(f'Epoch {self.epoch + 1} of {self.total_epochs}')


class Loss(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.loss = None

    def extract_metric(self, local_variables: dict):
        self.loss = self.get_variable(local_variables, 'loss').item()

    def print_metric(self):
        print(f'Loss: {self.loss}')

