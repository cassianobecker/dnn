from fwk.metrics import Metric
from fwk.config import Config


class SubjectTotals(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_subjects = None
        self.regime = None

    def extract_metric(self, local_variables: dict):
        self.number_of_subjects = len(local_variables['self'].data_loaders[self.regime].dataset.subjects)

    def print_metric(self):
        print(f'Number of {self.regime} subjects: {self.number_of_subjects}')


class TrainSubjectsTotals(SubjectTotals):

    def __init__(self) -> None:
        super().__init__()
        self.regime = 'train'


class TestSubjectsTotals(SubjectTotals):

    def __init__(self) -> None:
        super().__init__()
        self.regime = 'test'


class BatchCounter(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_subjects = None
        self.regime = None
        self.batch_idx = None
        self.subjects_per_batch = None
        self.number_of_batches = None

    def extract_metric(self, local_variables: dict):
        self.number_of_subjects = len(local_variables['data_loaders'][self.regime].dataset.subjects)
        self.batch_idx = local_variables['batch_idx']
        self.number_of_batches = int(self.number_of_subjects/self.subjects_per_batch)

    def print_metric(self):
        print(f'Batch {self.batch_idx + 1} of {self.number_of_batches} {self.regime} batches')


class TrainBatchCounter(BatchCounter):

    def __init__(self) -> None:
        super().__init__()
        self.regime = 'train'
        self.subjects_per_batch = int(Config.config['ALGORITHM']['train_batch_size'])


class TestBatchCounter(BatchCounter):

    def __init__(self) -> None:
        super().__init__()
        self.regime = 'test'
        self.subjects_per_batch = int(Config.config['ALGORITHM']['test_batch_size'])