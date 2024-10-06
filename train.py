import pandas as pd, numpy as np, matplotlib.pyplot as plt
import argparse

from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder

import time
from tqdm import tqdm


import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.init as init

from feature_engineering import replace_abnormal, add_features
from datasets import ObsDataset, TestDataset, get_dataloaders
from models import biLSTMClassifier
from utils import get_device, init_rnn, EarlyStopper

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('--train_datapath', default='data/X_train_N1UvY30.csv', type=str)
    parser.add_argument('--target_datapath', default='data/y_train_or6m3Ta.csv', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (default: %(default)s)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('checkpoints_path', type=st )
    
def train_loop(dataloader, model, loss_fn, optimizer, shortcut=0):
    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)

    train_loss = 0
    
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        
        loss = loss_fn(out, y)
        train_loss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if out.std() < 0.000001:
            print("WARNING: std() is zero, stopping")
            break

        if shortcut > 0 and batch == shortcut:
            return train_loss.detach().cpu().numpy() / shortcut
    return train_loss.detach().cpu().numpy() / num_batches


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            test_loss += loss_fn(out, y).detach().cpu().numpy()
    
        # scheduler.step(test_loss)
    return test_loss / num_batches

def main(args):
    # Load data
    raw_train_data = pd.read_csv(args.train_datapath)
    raw_target_data = pd.read_csv(args.target_datapath)

    # Init device
    device = get_device()
    print(f"Using {device}.")

    # Feature engineering
    train_data = replace_abnormal(raw_train_data)
    target_data = raw_target_data
    train_data, features = add_features(train_data) 

    # Features selection
    x_cols = features
    train_dataloader, test_dataloader = get_dataloaders(train_data, target_data, x_cols, args.batch_size)

    # Model
    if args.model == "biLSTM":
        LSTM_params = {
            "input_size":len(x_cols),
            "hidden_size":64,
            "num_layers":2,
            "bidirectional":True,
            "dropout":0,
        }

        MLP_params = {
            "in_features":LSTM_params["hidden_size"],
            "n_layers":2, # max 3
            "out_features":24,
        }
        model = biLSTMClassifier(LSTM_params, MLP_params)
    else:
        raise NotImplementedError("This model isn't implemented")
    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameters:", n_parameters)

    # Training hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5, verbose=True)
    early_stopper = EarlyStopper(patience=15, min_delta=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    history = pd.DataFrame([], columns=["epoch","train_loss","test_loss","lr"])

    for epoch in range(args.epochs):
        
        print(f"Epoch {epoch+1:>3d}",end=" ")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, shortcut=0)
        print(f"Train: {train_loss:>5f}", end=" ")
        test_loss = test_loop(test_dataloader, model, loss_fn)
        print(f"| Test: {test_loss:>5f}")

        if early_stopper.early_stop(test_loss, model):  
            model = early_stopper.get_best_model()
            break
        history.loc[len(history),:] = [epoch+1, train_loss, test_loss, optimizer.param_groups[0]['lr']]

    history[["train_loss", "test_loss"]].plot()

if __name__ == "__main__":
    args = get_args()
    main(args)
