import numpy as np
import sklearn.metrics

from fwk.metrics import Metric
from util.encode import one_hot_to_int

class ClassificationAccuracy(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.stats = None

    def on_after_train_batch(self, local_variables):
        self._on_after_batch(local_variables, 'train')

    def on_after_test_batch(self, local_variables):
        self._on_after_batch(local_variables, 'test')

    def _on_after_batch(self, local_variables, regime):

        outputs = local_variables['outputs'].cpu()
        targets = local_variables['targets'].cpu()

        predicted = outputs.argmax(dim=1).cpu()

        if regime not in self.stats.keys():
            self.stats[regime] = ClassificationStatistics()

        self.stats[regime].correct += predicted.eq(one_hot_to_int(targets)).sum().item()
        self.stats[regime].total += predicted.shape[0]

        self.stats[regime].predicted.extend(predicted.tolist())
        self.stats[regime].targets.extend(targets.tolist())

    def on_before_epoch(self, local_variables):
        self.stats = dict()

    def on_after_epoch(self, local_variables):

        for regime in self.stats.keys():

            self.stats[regime].accuracy = self.stats[regime].correct/self.stats[regime].total

            self.stats[regime].confusion_matrix = sklearn.metrics.confusion_matrix(
                self.stats[regime].targets, self.stats[regime].predicted)

            self.stats[regime].epoch = local_variables['epoch']

        self.print_metric()

    def text_record(self):

        text_record = ''

        for regime in self.stats.keys():

            accuracy_str = f'\n{regime} accuracy:\n{100*self.stats[regime].accuracy:.2f}% ' \
                           f'({self.stats[regime].correct} of {self.stats[regime].total})\n'

            confusion_str = f'\n{regime} confusion matrix:\n{np.array_str(self.stats[regime].confusion_matrix)}\n'

            text_record += accuracy_str + confusion_str + '\n'

        return text_record

    def numpy_record(self, records=None):

        for regime in self.stats.keys():

            if regime not in records.keys():
                records[regime] = dict()

            if 'accuracy' not in records[regime].keys():
                records[regime]['accuracy'] = list()

            records[regime]['accuracy'].append(self.stats[regime].accuracy)

            if 'confusion_matrix' not in records[regime].keys():
                records[regime]['confusion_matrix'] = list()

            records[regime]['confusion_matrix'].append(self.stats[regime].confusion_matrix)

        return records


class ClassificationStatistics:

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0
        self.predicted = []
        self.targets = []
        self.accuracy = None
        self.confusion_matrix = None
        self.epoch = None
