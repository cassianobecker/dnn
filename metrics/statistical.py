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
        self.accuracy = None
        self.confusion_matrix = None
        self.epoch = None

    def on_after_test_batch(self, local_variables):
        outputs = local_variables['outputs'].cpu()
        targets = local_variables['targets'].cpu()

        predicted = outputs.argmax(dim=1, keepdim=True).cpu()
        self.correct += predicted.eq(targets.view_as(predicted)).sum().item()
        self.total += predicted.shape[0]
        self.predicted.extend(list(np.squeeze(predicted.numpy())))
        self.targets.extend(list(targets.numpy()))

    def on_before_epoch(self, local_variables):
        self.correct = 0
        self.total = 0
        self.predicted = []
        self.targets = []

    def on_after_epoch(self, local_variables):
        self.accuracy = self.correct/self.total
        self.confusion_matrix = sklearn.metrics.confusion_matrix(self.targets, self.predicted)
        self.epoch = local_variables['epoch']
        self.print_metric()

    def text_record(self):
        accuracy_str = f'\naccuracy:\n{100*self.accuracy:.2f}% ({self.correct} of {self.total})\n'
        confusion_str = f'\nconfusion matrix:\n{np.array_str(self.confusion_matrix)}\n'
        return accuracy_str + confusion_str

    def numpy_record(self, records=None):

        if 'accuracy' not in records.keys():
            records['accuracy'] = list()

        records['accuracy'].append(self.accuracy)

        if 'confusion_matrix' not in records.keys():
            records['confusion_matrix'] = list()

        records['confusion_matrix'].append(self.confusion_matrix)

        return records
