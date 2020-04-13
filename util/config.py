import configparser
from itertools import product
import os
from datetime import datetime


class Config:

    config = None

    @staticmethod
    def set_config_from_url(config_url):
        config = configparser.ConfigParser()
        config.read(config_url)
        Config.config = config

    @staticmethod
    def set_config(config):
        Config.config = config

    @staticmethod
    def get_config():
        return Config.config


class ConfigProductGenerator:

    def __init__(self, config_url):

        config = configparser.ConfigParser()

        if os.path.isfile(config_url):
            config.read(config_url)
        else:
            raise Exception(f'ConfigParser file not found on {config_url}')

        self.config = config

        self.config_products = []
        self.config_product_urls = []

        self._config_products_path = None

        self.generate_config_products()

    def generate_config_products(self):

        base_config, all_config_lists = self._parse_config()
        self._create_config_products(base_config, all_config_lists)

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

        for (k, config_key_combinations) in enumerate(product(*all_config_lists)):

            config_product = configparser.ConfigParser()

            for config_for_key in config_key_combinations:
                config_product.read_dict(config_for_key)

            config_product.read_dict(base_config)

            self.config_products.append(config_product)

    def save_config_products(self,  experiment_name):

        self._config_products_path = self._make_config_paths(experiment_name)

        for (k, config_product) in enumerate(self.config_products):

            file_name = f'{k + 1}' + '.ini'
            file_url = os.path.join(self._config_products_path, file_name)

            self.config_product_urls.append(file_url)

            with open(file_url, 'w') as config_file:
                config_product.write(config_file)

    def _make_config_paths(self, experiment_name):

        results_base_path = self.config['OUTPUTS']['base_path']

        if not os.path.exists(results_base_path):
            raise FileNotFoundError(f"Top folder for results: '{results_base_path}' does no exist")


        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(results_base_path, experiment_name, current_time)
        config_products_path = os.path.join(results_path, 'config_products')

        os.makedirs(config_products_path)

        return config_products_path


def _test_save_configs():

    config_url = 'test/args.ini'
    config_generator = ConfigProductGenerator(config_url)
    config_generator.save_config_products(experiment_name='dnn_hcp')

    config_product_index = 4
    config_product_url = config_generator.config_product_urls[config_product_index]
    Config.set_config_from_url(config_product_url)

    pass


if __name__ == '__main__':
    _test_save_configs()

