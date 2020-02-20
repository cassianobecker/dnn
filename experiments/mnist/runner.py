import os
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score
from util.logging import set_logger, get_logger
from util.experiment import print_memory


class Runner:

    def __init__(self, device, params, train_loader, test_loader):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        log_furl = os.path.join(params['FILE']['experiment_path'], 'log', 'experiment.log')
        set_logger('Experiment', params['LOGGING']['experiment_level'], log_furl)
        self.experiment_logger = get_logger('Experiment')

        log_furl = os.path.join(params['FILE']['experiment_path'], 'log', 'monitor.log')
        set_logger('Monitor', params['LOGGING']['monitor_level'], log_furl)
        self.monitor_logger = get_logger('Monitor')

        self.monitor_logger.info('creating runner class')

        self.name = params['FILE']['experiment_name']
        self.path = params['FILE']['experiment_path']

        k = 1.
        self.w = torch.tensor([1., k, k, k, k, k]).to(self.device)

    def save_model(self, model, epoch='last'):

        model_furl = self._get_model_furl(epoch)
        self.monitor_logger.info('Saving model at: {:} for epoch {:}.'.format(model_furl, epoch))
        torch.save(model.state_dict(), model_furl)

    def load_model(self, model, epoch='last'):

        model_furl = self._get_model_furl(epoch)
        self.monitor_logger.info('Loading model from: {:} for epoch {:}.'.format(model_furl, epoch))
        model_pars = torch.load('{:}'.format(model_furl))
        model.load_state_dict(model_pars)

        return model

    def initial_save_and_load(self, model, restart=False):
        # saves first iteration
        self.save_model(model, 0)

        # also save first iteration as last, if no previous higher iterations have been saved
        if self._check_saved_model(epoch='last') is False or restart is True:
            self.save_model(model, epoch='last')

        # loads last iteration saved, if any
        model = self.load_model(model)
        return model

    def _check_saved_model(self, epoch='last'):

        model_furl = self._get_model_furl(epoch)
        exists = os.path.exists(model_furl)

        return exists

    def _get_model_furl(self, epoch):
        model_furl = os.path.join(self.path, 'out', 'model_epoch_{:}.pt'.format(epoch))
        return model_furl

    def run(self, args, model, run_initial_test=True):

        mini_batch = 10

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        self.monitor_logger.info(' ***** starting experiment *****')

        if run_initial_test is True:
            test_loss_value, predictions, targets = self.test_batch(model)

            self.print_eval(test_loss_value, predictions, targets, idx=0, header='Test epoch:')
            self.print_confusion_matrix(predictions, targets)

        for epoch in range(1, args.epochs + 1):

            train_loss_value, predictions, targets = self.train_minibatch(model, optimizer, mini_batch)

            self.print_eval(train_loss_value, predictions, targets, idx=epoch, header='Train epoch:')
            self.print_confusion_matrix(predictions, targets)

            scheduler.step()

            if args.save_model:
                self.save_model(model, epoch)
                self.save_model(model)

            test_loss_value, predictions, targets = self.test_batch(model)

            self.print_eval(test_loss_value, predictions, targets, idx=epoch, header='Test epoch:')
            self.print_confusion_matrix(predictions, targets)

    def train_minibatch(self, model, optimizer, mini_batch=10):
        """
        Loads input data (BOLD signal windows and corresponding target motor tasks) from one patient at a time,
        and minibatches the windowed input signal while training the TGCN by optimizing for minimal training loss.
        :return: train_loss
        """
        model.train()

        train_loss_value = 0
        predictions = []
        targets = []

        for batch_idx, (data, target) in enumerate(self.train_loader):

            self.monitor_logger.info('training on batch idx  {:}'.format(batch_idx + 1))

            self.monitor_logger.info(print_memory())

            batch_loss_value = 0
            batch_predictions = []
            batch_targets = []

            output = model(data)
            # prediction = output.max(1, keepdim=True)[1][0]
            prediction = output.argmax(dim=1, keepdim=True)

            torch.cuda.synchronize()

            loss = F.nll_loss(output, target, weight=self.w)
            loss = loss / mini_batch
            loss.backward()

            train_loss_value += loss.item()
            predictions.extend(prediction.tolist())
            targets.extend(target.tolist())

            batch_loss_value += loss.item()
            batch_predictions.extend(prediction.tolist())
            batch_targets.extend(target.tolist())

            self.print_eval(batch_loss_value, batch_predictions, batch_targets, idx=batch_idx, header='batch idx:')

        return train_loss_value, predictions, targets

    def test_batch(self, model):
        """
        Evaluates the model trained in train_minibatch() on patients loaded from the test set.
        :return: test_loss and correct, the # of correct predictions
        """
        model.eval()

        test_loss_value = 0
        predictions = []
        targets = []

        with torch.no_grad():

            for data, target in self.test_loader:

                self.monitor_logger.info(print_memory())

                data, target = data.to(self.device), target.to(self.device)
                output = model(data)

                prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                torch.cuda.synchronize()

                test_loss_value += F.nll_loss(output, target, reduction='sum').item()

                predictions.extend(prediction.tolist())
                targets.extend(target.tolist())

        return test_loss_value, predictions, targets

    def print_eval(self, loss_value, predictions, targets, idx=None, header=''):

        accuracy = accuracy_score(targets, predictions)

        msg = f'| {header:14s}' + f'{idx:3d} | '
        msg = msg + f'loss: {loss_value:1.3e} | '
        msg = msg + f'accuracy: {accuracy: 1.3f} |'

        self.experiment_logger.info(msg)
        print(msg)

    def print_confusion_matrix(self, predictions, targets):

        cm = confusion_matrix(targets, predictions)
        self.experiment_logger.info(cm)
        print(cm)
