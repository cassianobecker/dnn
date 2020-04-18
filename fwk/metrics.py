from util.lang import class_for_name
from util.path import absolute_path
from util.string import list_from
from fwk.config import Config
from fwk.observers import Observer
import configparser


class Metric:

    def __init__(self) -> None:
        self.observers = list()

    def add_observer(self, channel):
        self.observers.append(channel)

    def get_implemented_events(self):
        event_handler_list = [m[3:] for m in self.__class__.__dict__ if m.startswith('on_')]
        return event_handler_list

    def on_event(self, event_type, local_variables):
        event_handler = method_to_call = getattr(self, 'on_' + event_type)
        event_handler(local_variables)

    def print_metric(self):
        for observer in self.observers:
            observer.print(self)

    def on_before_setup(self, local_variables):
        pass

    def on_after_setup(self, local_variables):
        pass

    def on_before_epoch(self, local_variables):
        pass

    def on_after_epoch(self, local_variables):
        pass

    def on_before_train_batch(self, local_variables):
        pass

    def on_after_train_batch(self, local_variables):
        pass

    def on_before_test_batch(self, local_variables):
        pass

    def on_after_test_batch(self, local_variables):
        pass


class MetricsHandler:

    metrics_for_events = dict()

    @classmethod
    def register_metrics(cls):

        metrics_config, observers_config = cls._get_configs()

        observers = dict()

        for metrics_class_name in metrics_config.sections():

            metric = class_for_name(metrics_class_name)()

            for observer_name in list_from(metrics_config[metrics_class_name]['observers']):

                if observer_name not in observers.keys():
                    observer = cls._create_observer(observers_config, observer_name)
                    observers[observer_name] = observer
                else:
                    observer = observers[observer_name]

                metric.add_observer(observer)

            cls.register_events(metric)

        pass

    @classmethod
    def register_events(cls, metric):
        for event in metric.get_implemented_events():
            if event not in cls.metrics_for_events:
                cls.metrics_for_events[event] = list()

            cls.metrics_for_events[event].append(metric)

    @classmethod
    def _create_observer(cls, observers_config, observer_name):

        observer = Observer()

        renderer_class_names = list_from(observers_config[observer_name]['renderers'])
        for renderers_class_name in renderer_class_names:
            renderer = class_for_name(renderers_class_name)(observer_name)
            observer.add_renderer(renderer)

        return observer

    @classmethod
    def _get_configs(cls):

        # optionxform preserves CamelCase in keys for loading class names
        # see https://docs.python.org/2/library/configparser.html#ConfigParser.RawConfigParser.optionxform

        metrics_config = configparser.ConfigParser(allow_no_value=True)
        metrics_config.optionxform = str
        metrics_ini_url = Config.config['METRICS']['metrics_ini_url']
        metrics_config.read(absolute_path(metrics_ini_url))

        observers_config = configparser.ConfigParser(allow_no_value=True)
        observers_ini_url = Config.config['METRICS']['observers_ini_url']
        observers_config.optionxform = str
        observers_config.read(absolute_path(observers_ini_url))

        return metrics_config, observers_config

    @classmethod
    def dispatch_event(cls, local_variables, event_type):

        if event_type in cls.metrics_for_events.keys():
            for metric in cls.metrics_for_events[event_type]:
                metric.on_event(event_type, local_variables)
