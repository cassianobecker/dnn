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


class SubjectList(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.train_subjects = None
        self.test_subjects = None
        self.regime = None

    def on_after_setup(self, local_variables):
        self.train_subjects = local_variables['self'].data_loaders['train'].dataset.subjects
        self.test_subjects = local_variables['self'].data_loaders['test'].dataset.subjects
        self.print_metric()

    def text_record(self):

        train_str = f'TRAIN:\n'
        train_str = train_str + '\n'.join(self.train_subjects)

        test_str = f'TEST:\n'
        test_str = test_str + '\n'.join(self.test_subjects)

        return train_str + '\n\n' + test_str


class ImageTotals(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_train_images = None
        self.number_of_test_images = None
        self.regime = None

    def on_after_setup(self, local_variables):
        self.number_of_train_images = len(local_variables['self'].data_loaders['train'].dataset.images)
        self.number_of_test_images = len(local_variables['self'].data_loaders['test'].dataset.images)
        self.print_metric()

    def text_record(self):
        train_str = f'number of subjects: {self.number_of_train_images} (train)\n'
        test_str = f'number of subjects: {self.number_of_test_images} (test)\n'
        return train_str + test_str

    def numpy_record(self, records=None):

        if 'number_of_train_subjects' not in records.keys():
            records['number_of_train_subjects'] = self.number_of_train_images

        if 'number_of_test_subjects' not in records.keys():
            records['number_of_test_subjects'] = self.number_of_test_images

        return records


class ImageList(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.train_images = None
        self.test_images = None
        self.regime = None

    def on_after_setup(self, local_variables):
        self.train_images = local_variables['self'].data_loaders['train'].dataset.images
        self.test_images = local_variables['self'].data_loaders['test'].dataset.images
        self.print_metric()

    def text_record(self):

        train_str = f'TRAIN:\n'
        train_str = train_str + '\n'.join(self.train_images)

        test_str = f'TEST:\n'
        test_str = test_str + '\n'.join(self.test_images)

        return train_str + '\n\n' + test_str
