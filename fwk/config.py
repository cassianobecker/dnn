import configparser
from itertools import product
import os
from datetime import datetime
from util.string import range_to_comma_separated_string


class Config:

    config: configparser = None

    @classmethod
    def set_config_from_url(cls, config_url):
        config = configparser.ConfigParser()
        config.read(config_url)
        cls.config = config


    @classmethod
    def set_config(cls, config):
        cls.config = config

    @classmethod
    def get_config(cls):
        return cls.config


class ConfigProductGenerator:

    def __init__(self, config_url):

        config = configparser.ConfigParser()

        if os.path.isfile(config_url):
            config.read(config_url)
        else:
            raise Exception(f'ConfigParser file not found on {config_url}')

        self.config = config
        self._pre_process_config()

        self.config_products = []
        self.config_product_urls = []
        self.results_path = None
        self.generate_config_products()

    def generate_config_products(self):

        base_config, all_config_lists = self._parse_config()
        self._create_config_products(base_config, all_config_lists)

    def _add_subject_batches(self, session, key):

        if session == 'SUBJECTS' and key == 'number_of_batches':
            number_of_batches = int(self.config['SUBJECTS']['number_of_batches'])
            self.config['SUBJECTS']['subject_batch_index'] = range_to_comma_separated_string(number_of_batches)

    def _pre_process_config(self):

        pre_processing_functions = [self._add_subject_batches]

        for session in self.config.keys():
            for key in self.config[session]:
                for pre_process_function in pre_processing_functions:
                    pre_process_function(session, key)

    def has_multiple_products(self):
        return len(self.config_products) > 1

    def _parse_config(self):

        base_config = configparser.ConfigParser()
        all_config_lists = []

        for session in self.config.keys():

            for key in self.config[session]:
                values = self._break_values(self.config[session][key])

                if len(values) > 1:
                    config_list = self._list_of_configs_for_values(session, key, values)
                    all_config_lists.append(config_list)

                else:
                    if not base_config.has_section(session):
                        base_config.add_section(session)

                    base_config[session][key] = values[0]

        return base_config, all_config_lists

    @staticmethod
    def _break_values(values):
        return [token.strip() for token in values.split(',')]

    @staticmethod
    def _list_of_configs_for_values(session, key, values):

        config_list = []
        for value in values:
            current_config = configparser.ConfigParser()
            if not current_config.has_section(session):
                current_config.add_section(session)
            current_config[session][key] = value
            config_list.append(current_config)

        return config_list

    def _create_config_products(self, base_config, all_config_lists):

        if len(all_config_lists) == 0:
            self.config_products.append(base_config)

        else:

            for (k, config_key_combinations) in enumerate(product(*all_config_lists)):

                config_product = configparser.ConfigParser()

                for config_for_key in config_key_combinations:
                    config_product.read_dict(config_for_key)

                config_product.read_dict(base_config)

                self.config_products.append(config_product)

    def save_config_products(self):

        self._make_results_path()

        for (k, config_product) in enumerate(self.config_products):

            config_product_path = self._make_config_product_path(k)

            file_name = f'args' + '.ini'
            file_url = os.path.join(config_product_path, file_name)

            self.config_product_urls.append(file_url)

            with open(file_url, 'w') as config_file:
                config_product.write(config_file)

    def _make_results_path(self):

        results_base_path = os.path.expanduser(self.config['OUTPUTS']['base_path'])
        experiment_short_name = self.config['EXPERIMENT']['short_name']

        os.makedirs(results_base_path, exist_ok=True)

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.results_path = os.path.join(results_base_path, experiment_short_name, current_time)

    def _make_config_product_path(self, k):

        config_product_path = os.path.join(self.results_path, f'config_product_{k + 1}')

        os.makedirs(config_product_path)

        return config_product_path


def _test_save_configs():

    config_url = 'test/args.ini'
    config_generator = ConfigProductGenerator(config_url)
    config_generator.save_config_products()

    config_product_index = 4
    config_product_url = config_generator.config_product_urls[config_product_index]
    Config.set_config_from_url(config_product_url)

    pass


if __name__ == '__main__':
    _test_save_configs()
