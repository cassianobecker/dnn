from fwk.metrics import Metric


class SubjectTotals(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_train_subjects = None
        self.number_of_test_subjects = None
        self.regime = None

    def on_after_setup(self, local_variables):
        self.number_of_train_subjects = len(local_variables['self'].data_loaders['train'].dataset.subjects)
        self.number_of_test_subjects = len(local_variables['self'].data_loaders['test'].dataset.subjects)
        self.print_metric()

    def text_record(self):
        train_str = f'number of subjects: {self.number_of_train_subjects} (train)\n'
        test_str = f'number of subjects: {self.number_of_test_subjects} (test)\n'
        return train_str + test_str

    def numpy_record(self, records=None):

        if 'number_of_train_subjects' not in records.keys():
            records['number_of_train_subjects'] = self.number_of_train_subjects

        if 'number_of_test_subjects' not in records.keys():
            records['number_of_test_subjects'] = self.number_of_test_subjects

        return records
