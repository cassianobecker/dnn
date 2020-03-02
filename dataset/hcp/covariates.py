import pandas as pd
import os, pathlib
from util.path import get_root


class Covariates:

    def __init__(self):

        self.remote_url = 'https://db.humanconnectome.org/REST/search/dict/Subject%20Information/results?format=csv&removeDelimitersFromFieldValues=true&restricted=0&project=HCP_1200'
        self.local_fname = 'hcp_covariates.csv'
        self.local_path = os.path.join(get_root(), 'dataset', 'hcp', 'res')
        self.local_url = os.path.join(self.local_path, self.local_fname)

        if os.path.isfile(self.local_url):
            self.df = pd.read_csv(self.local_url)
        else:

            error_str = '\nLocal file with HCP covariates information not found.\n' \
                        'Please download file from:\n{0}\n' \
                        'Then, rename it to "{1}" and save it to folder:\n{2}\n'\
                .format(self.remote_url, self.local_fname, pathlib.Path(self.local_path).as_uri())

            raise FileNotFoundError(error_str)

    def value(self, field, subject):
        row = self.df.loc[self.df['Subject'] == int(subject)]
        value = getattr(row, field).values[0]
        return value

    def fields(self):
        return list(self.df.columns)


def test_covariates():

    cov = Covariates()
    fields = cov.fields()
    print('All fields: {0}'.format(fields))

    subject = '100307'
    field = fields[3]
    value = cov.value(field, subject)
    print('{0} value for subject {1}: {2}'.format(field, subject, value))

    pass


if __name__ == '__main__':
    test_covariates()
