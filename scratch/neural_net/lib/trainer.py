import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from IPython import embed




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=300, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter % 200 == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class Trainer:
    def __init__(self, train_loader, test_loader, Optimizertype=optim.Adam, optim_kwargs=None, learning_rate=0.01, epochs=1000, n_patience=None, break_out=None, log_interval=100)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.break_out = break_out
        self.log_interval = log_interval

        self.stopper = None
        # n_patience is number of epochs to go over where test CV increases before breaking out to avoid overfitting
        if n_patience is not None:
            self.stopper = EarlyStopping(patience=n_patience*len(train_loader))

        if not issubclass(optimizertype, optim.Optimizer):
            raise ValueError("Supplied optimizer type ({}) is incorrect".format(optimizertype))

        self.Optimizertype = Optimizertype
        self.optim_kwargs = optim_kwargs

        self.losses_train = None
        self.losses_test = None


    # Dimension of input features
    @property
    def n_dim(self):
        return self.train_loader.dataset[0][0].shape[1:]

    # Total number of samples in training dataset
    @property
    def n_data(self):
        return len(self.train_loader.dataset)

    # Number of samples per training batch
    @property
    def batch_size(self):
        return self.train_loader.batch_size

    # Number of training batches per epoch
    @property
    def n_batches(self):
        return len(self.train_loader)

    # Number of training rounds (batches) per epoch
    @property
    def n_steps(self):
        return self.epochs * self.n_batches


    
    # Train net based on some loss criterion
    #   Optionally supply function to transform net output before calculating loss
    def train(self, net, criterion, loss_fn=None, loss_fn_args=None):

        optimizer = self.Optimizertype(net.parameters(), lr=self.learning_rate, **self.optim_kwargs)
        
        # Initialize losses - save loss after each training batch
        self.losses_train = np.zeros(self.n_steps)
        self.losses_test = np.zeros_like(losses_train)
        
        idx = 0

        for epoch in range(epochs):

            for batch_idx, (data, target) in enumerate(train_loader):

                if not do_cnn:
                    # resize data from (batch_size, 1, n_input_dim) to (batch_size, n_input_dim)  
                    data = data.view(-1, n_dim)
                net_out = net(data)
                
                if loss_fn is None:
                    loss = criterion(net_out, target)
                else:
                    loss = loss_fn(net_out, target, criterion, *loss_fn_args)
                #losses[batch_idx + epoch*n_batches] = loss.item()

                # Back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### VALIDATION ###
                data_test, target_test = iter(test_loader).next()
                if not do_cnn:
                    data_test = data_test.view(-1, n_dim)
                test_out = net(data_test).detach()
                if loss_fn is None:
                    test_loss = criterion(test_out, target_test).item()
                else:
                    test_loss = loss_fn(test_out, target_test, criterion, *loss_fn_args)

                stopper(test_loss, net)
                if stopper.early_stop:
                    print("Breaking out of training loop")
                    return losses_train, losses_test

                if epoch % log_interval == 0:

                    losses_train[idx] = loss.item()
                    losses_test[idx] = test_loss

                    idx += 1

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (valid: {:.6f}) (diff: {:.6f})'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item(), test_loss, test_loss-loss.item()))
                    if break_out is not None and test_loss < break_out:
                        print("test loss is lower than break out; breaking out of loop")
                        return losses_train, losses_test


        return losses_train, losses_test


