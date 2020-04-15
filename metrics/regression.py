from fwk.metrics import Metric

class SomeMetrics(Metric):

    def __init__(self) -> None:
        super().__init__()

    def extract_metric(self, local_variables: dict):

        train_set = self.get_variable(local_variables, 'train_set')
        self.number_of_train_subjects = len(train_set.subjects)


    def print_metric(self):