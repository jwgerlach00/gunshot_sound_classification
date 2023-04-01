import torch
from torch.utils.data import DataLoader
from helpers.preprocessing import cross_entropy_weights, get_distribution, normalize_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List
from copy import deepcopy
import numpy as np


class TrainingLoop:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, ModelArchitecture:torch.nn.Module, Dataset:torch.utils.data.Dataset, hyperparams):
        print(f'Using device: {TrainingLoop.device}')
        
        # Set instance methods
        self.plot_loss = self._plot_loss
        self.save_model = self._save_model
        
        self.hyperparams = deepcopy(hyperparams)
        self.Dataset = Dataset
        self.model = ModelArchitecture(self.hyperparams).to(TrainingLoop.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['learning_rate'])
        self.criterion = torch.nn.MSELoss()
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc = None
        self.train_acc = None
    
    @staticmethod
    def dataloader(Dataset:torch.utils.data.Dataset, hyperparams:dict,size:int) -> DataLoader:
        return DataLoader(Dataset(hyperparams,size), batch_size=hyperparams['batch_size'])

    def training_loop(self,):
        # Dataloaders
        train_generator = TrainingLoop.dataloader(self.Dataset, self.hyperparams,4)
        val_generator = TrainingLoop.dataloader(self.Dataset, self.hyperparams,1)
        for epoch in range(1, self.hyperparams['epochs'] + 1):
            print(f'Epoch {epoch}')
            
            # # Normalization (optionally)
            # if self.hyperparams['normalize']['run']:
            #     X_train, scaler = normalize_data(X_train, method=self.hyperparams['normalize']['method'])
            #     if scaler: # None if method is 'mean'
            #         X_val = scaler.transform(X_val)
            
            # Batch train
            self.model.train()
            batch_train_loss_history = []
            for (X, y) in tqdm(train_generator):
                self.optimizer.zero_grad()
                
                y_p = self.model(X)
                loss = self.criterion(y_p.squeeze(),y)

                loss.backward()
                self.optimizer.step()
                batch_train_loss_history.append(loss.item())
            
            # Batch validation
            self.model.eval()
            batch_val_loss_history = []
            for (X, y) in tqdm(val_generator):
                with torch.no_grad():
                    y_p = self.model(X)
                
                loss = self.criterion(y_p.squeeze(), y)
                batch_val_loss_history.append(loss.item())
            
            # Batch average loss
            epoch_train_loss = sum(batch_train_loss_history) / len(batch_train_loss_history)
            epoch_val_loss = sum(batch_val_loss_history) / len(batch_val_loss_history)
            print(f'Train loss: {epoch_train_loss:.4f}\nVal loss: {epoch_val_loss:.4f}')
            
            # Append batch loss to epoch loss list
            self.train_loss_history.append(epoch_train_loss)
            self.val_loss_history.append(epoch_val_loss)
        
        # Calculate accuracy
        #self.train_acc = TrainingLoop.accuracy(self.model, train_generator)
        #self.val_acc = TrainingLoop.accuracy(self.model, val_generator)
  
        return self.model

    @staticmethod
    def plot_loss(train_loss_history:List[float], val_loss_history:List[float], hyperparams:dict) -> None:
        plt.figure()
        plt.title('Loss curve')
        plt.plot(range(len(train_loss_history)), train_loss_history, label='train loss')
        plt.plot(range(len(val_loss_history)), val_loss_history, label='val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def _plot_loss(self) -> None:
        TrainingLoop.plot_loss(self.train_loss_history, self.val_loss_history, self.hyperparams)

    @staticmethod
    def save_model(model, path:str) -> None:
        torch.save(model.state_dict(), path)
        
    def _save_model(self, path:str) -> None:
        TrainingLoop.save_model(self.model, path)
    
    @staticmethod
    def accuracy(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader) -> float:
        sum = 0
        length = 0
        for (X, y) in tqdm(dataloader):
            model.eval()
            with torch.no_grad():
                y_p = model(X)
                sum += torch.sum(y == torch.round(y_p)).item()
                length += len(y_p)
        return sum/length
