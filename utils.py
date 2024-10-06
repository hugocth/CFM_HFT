import math

import torch
import torch.nn.init as init


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
    
def init_rnn(x, type='uniform'):
    for layer in x._all_weights:
        for w in layer:
            if 'weight' in w:
                if type == 'xavier':
                    init.xavier_normal_(getattr(x, w))
                elif type == 'uniform':
                    stdv = 1.0 / (getattr(x, w).size(-1))**0.5
                    init.uniform_(getattr(x, w), -stdv, stdv)
                elif type == 'normal':
                    stdv = 1.0 / (getattr(x, w).size(-1))**0.5
                    init.normal_(getattr(x, w), 0.0, stdv)
                else:
                    raise ValueError
                
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.00001, verbose=True):
        self.best_model = None
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_acc = 0
        self.verbose = verbose
        
    def get_best_model(self):
        return self.best_model

    def early_stop(self, validation_loss, acc, model):
        if validation_loss < self.min_validation_loss:
            if self.verbose:
                print(f"New best loss: {validation_loss:>4f}")
                print(f"Acc: {acc}")
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = model
            self.best_acc = acc
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False