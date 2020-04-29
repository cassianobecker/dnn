from dataset.hcp.subjects import Subjects
from fwk.config import Config
from util.path import append_path


def load_subject():

    train_subjects, test_subjects = Subjects().create_list_from_config()

    print(f'Number of train: {len(train_subjects)}; Number of test: {len(test_subjects)}')

    pass


if __name__ == '__main__':
    Config.set_config_from_url(append_path(__file__, 'conf/args.ini'))
    load_subject()
