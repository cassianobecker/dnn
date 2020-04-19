import torch
from metrics.dataset import load_mnist
import torch.utils.data.dataset


class Dataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index].astype('float32')
        y = self.labels[index].astype('float32')

        return x, y


def loaders(train_size, test_size, batch_size):
    _, train_data, train_labels, test_data, test_labels = load_mnist()

    train_data = train_data[:train_size, :]
    train_labels = train_labels[:train_size, :]
    test_data = test_data[:test_size, :]
    test_labels = test_labels[:test_size, :]

    training_set = Dataset(train_data, train_labels)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

    validation_set = Dataset(test_data, test_labels)

    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)

    return train_loader, test_loader
