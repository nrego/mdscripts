import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
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

## wrapper to handle all training shenanigans
#    Given the training and testing datasets, and all training parameters
#      Will run for specified number of epochs and keep track of training and validation performance
class Trainer:

    def __init__(self, train_dataset, test_dataset, batch_size=200, shuffle=True,
                 Optimizertype=optim.Adam, optim_kwargs={}, learning_rate=0.001, 
                 epochs=1000, n_patience=None, break_out=None, log_interval=100):

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        self.test_X, self.test_y = iter(test_loader).next()
        if torch.cuda.is_available():
            self.test_X = self.test_X.cuda()
            self.test_y = self.test_y.cuda()

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.break_out = break_out
        self.log_interval = log_interval

        self.stopper = None
        # n_patience is number of epochs to go over where test CV increases before breaking out to avoid overfitting
        if n_patience is not None:
            self.stopper = EarlyStopping(patience=n_patience*len(self.train_loader))

        if not issubclass(Optimizertype, optim.Optimizer):
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
    def __call__(self, net, criterion, loss_fn=None, loss_fn_kwargs={}):

        optimizer = self.Optimizertype(net.parameters(), lr=self.learning_rate, **self.optim_kwargs)
        
        # Initialize losses - save loss after each training batch
        self.losses_train = np.zeros(self.n_steps)
        self.losses_train[:] = np.inf
        self.losses_test = np.zeros_like(self.losses_train)
        self.losses_test[:] = np.inf
        
        for epoch in range(self.epochs):

            for batch_idx, (train_X, train_y) in enumerate(self.train_loader):

                if torch.cuda.is_available():
                    train_X = train_X.cuda()
                    train_y = train_y.cuda()
                    
                idx = epoch*self.n_batches + batch_idx
                ### TRAIN THIS BATCH ###

                net_out = net(train_X)
                
                if loss_fn is None:
                    train_loss = criterion(net_out, train_y)
                else:
                    train_loss = loss_fn(net_out, train_y, criterion, **loss_fn_kwargs)
                self.losses_train[idx] = train_loss.item()

                # Back prop
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                ### VALIDATION ###
                test_out = net(self.test_X).detach()
                if loss_fn is None:
                    test_loss = criterion(test_out, self.test_y).item()
                else:
                    test_loss = loss_fn(test_out, self.test_y, criterion, **loss_fn_kwargs)

                self.losses_test[idx] = test_loss
                
                ## Optionally break out of training
                if self.break_out is not None and test_loss < self.break_out:
                    blah = net(self.test_X).detach()
                    print("test loss ({:.2f}) is lower than break out ({:.2f}); breaking out of loop".format(test_loss, self.break_out))
                    return 

                if self.stopper is not None:
                    self.stopper(test_loss, net)

                    if self.stopper.early_stop:
                        print("Breaking out of training loop")
                        return 

                if epoch % self.log_interval == 0:
                    outstr = 'Train Epoch: {} '\
                             '[{}/{} ({:0.0f}%)]'\
                             '    Loss: {:0.6f}'\
                             '  (valid: {:0.6f})'.format(epoch, batch_idx * len(train_y), self.n_data, 
                                                         100*batch_idx/self.n_batches, train_loss.item(), test_loss)
                    print(outstr)


        return 


