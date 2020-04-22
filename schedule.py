from fwk.config import ConfigProductGenerator
import sys
import os
import subprocess
from fwk.experiment import Experiment
from util.path import absolute_path


class LocalExperimentRunner:
    @staticmethod
    def run(config_url):
        print(f"Using 'local' runner for configuration {config_url}")
        process_str = f'fwk/shell/local.sh {config_url}'
        subprocess.Popen(process_str, shell=True)


class CbicaExperimentRunner:
    @staticmethod
    def run(config_url):
        print(f"Using 'cbica' runner for configuration {config_url}")
        log_url = os.path.join(os.path.dirname(config_url), 'log')
        process_str = f'fwk/shell/cbica_gpu.sh {config_url} {log_url}'
        subprocess.Popen(process_str, shell=True)


class DebugExperimentRunner:
    @staticmethod
    def run(config_url):
        print("Using 'debug' runner")
        Experiment.run(config_url)


class ExperimentScheduler:

    config_generator: ConfigProductGenerator = None

    runners = {
        'local': LocalExperimentRunner(),
        'cbica': CbicaExperimentRunner(),
        'debug': DebugExperimentRunner(),
    }

    runner = None

    @classmethod
    def list_runners(cls):
        return ''.join([key + ' ' for key in cls.runners.keys()])

    @classmethod
    def print_usage(cls, argv):

        print('Usage: python schedule <runner> <args.ini>')
        print(f'    implemented runners: {cls.list_runners()}')

    @classmethod
    def initialize_configs(cls, argv):

        if len(argv) < 3:
            cls.print_usage(argv)
            exit(1)

        execution_environment = argv[1]
        config_url = argv[2]

        cls.config_generator = ConfigProductGenerator(absolute_path(config_url))
        cls.config_generator.save_config_products()
        cls.runner = cls.runners[execution_environment]

    @classmethod
    def run(cls):
        config_product_url: str
        for (k, config_product_url) in enumerate(cls.config_generator.config_product_urls):
            print(f'\nDNN - Deep Diffusion Convolution Neural Networks\nScheduling configuration {config_product_url}')
            cls.runner.run(config_product_url)

        pass


if __name__ == '__main__':
    ExperimentScheduler.initialize_configs(sys.argv)
    ExperimentScheduler.run()
