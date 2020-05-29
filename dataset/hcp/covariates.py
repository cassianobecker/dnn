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

    def value(self, field, subject, regression=True):

        if regression is True:
            mean = self.df[field].mean()
            std = self.df[field].std()
            values = self.df[self.df['Subject'] == int(subject)][field].values[0]
            values = (values - mean) / std
        else:
            dfd = pd.get_dummies(self.df[field])
            df = pd.concat([self.df, dfd], axis=1)[['Subject'] + list(dfd.columns)]

            row = df.loc[df['Subject'] == int(subject)]
            values = row[dfd.columns].values[0]

        return values

    def column_names(self, field):
        return list(pd.get_dummies(self.df[field]).columns)

    def fields(self):
        return list(self.df.columns)


def test_covariates():

    cov = Covariates()
    fields = cov.fields()
    print('All fields: {0}'.format(fields))

    subject = '100307'
    field = fields[3]
    values = cov.value(field, subject)
    columns = cov.column_names(field)

    print('{0} values for subject {1}: {2} for columns {3} '.format(field, subject, values, columns))

    pass


if __name__ == '__main__':
    test_covariates()
