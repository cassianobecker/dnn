from fwk.metrics import Metric
import numpy as np
import sklearn.metrics


class ClassificationAccuracy(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.correct = None
        self.total = None
        self.predicted = None
        self.targets = None

    def extract_metric(self, local_variables: dict, event_type):

        if event_type == 'AFTER_TRAIN_BATCH':

            outputs = local_variables['outputs']
            targets = local_variables['targets']

            predicted = outputs.argmax(dim=1, keepdim=True)
            self.correct += predicted.eq(targets.view_as(predicted)).sum().item()
            self.total += predicted.shape[0]
            self.predicted.extend(list(np.squeeze(predicted.numpy())))
            self.targets.extend(list(targets.numpy()))

        if event_type == 'BEFORE_EPOCH':
            self.correct = 0
            self.total = 0
            self.predicted = []
            self.targets = []

    def print_metric(self, event_type):

        if event_type == 'AFTER_EPOCH':
            accuracy_str = f'==> accuracy: {100*self.correct/self.total:.2f}%, correct: {self.correct}, total: {self.total}'
            print(accuracy_str)
            print('==> confusion matrix:')
            print(sklearn.metrics.confusion_matrix(self.targets, self.predicted))


class ConfusionMatrix(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.variable = None

    def extract_metric(self, local_variables: dict, event_type):
        pass

    def print_metric(self, event_type):
        pass


class SpecificMetric(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.variable = None

    def extract_metric(self, local_variables: dict, event_type):

        argument = self.get_variable(local_variables, 'variable_name')
        self.variable = argument

    def print_metric(self, event_type):
        pass
