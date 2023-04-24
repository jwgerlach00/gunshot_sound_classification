import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import os

def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16)) 
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap=sns.diverging_palette(220, 20, n=7),
        ax=ax
    )


class ResNetModel(nn.Module):
    def __init__(self,x_shape, BATCH_SIZE):
        super(ResNetModel, self).__init__()
        self.batch = BATCH_SIZE
        self.input = nn.Linear(1000, 512)
        self.relu = nn.ReLU() # Activation function
        self.length = x_shape
        self.hidden_layer = nn.Linear(512, x_shape)
        self.output = nn.Sigmoid()
        self.rn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
    def forward(self, x):
        x = x.reshape((self.batch,3,self.length,683)) # x = x.reshape((self.batch,3,self.length,43))
        x = self.relu(self.input(self.rn(x)))
        return self.output(self.hidden_layer(x))

class Dataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray):
        self.X = X.copy()
        self.y = y.copy()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx:int):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx]).to(torch.float32)
        )
        
def calc_acc(y, y_p):
    return torch.sum((y_p > 0.5).int() == y) / (y.shape[0] * y.shape[1])

def zeros_and_ones(t):
    ones = torch.tensor((t*2),dtype=torch.long).sum().detach().item()
    total = len(t.detach().numpy())
    return (total-ones)/total*100,ones/total*100

def class_counts(y):
    return [torch.sum(y == 0), torch.sum(y == 1)]

def distribution(y):
    frac_ones = np.sum(y) / (y.shape[0] * y.shape[1])
    return torch.tensor([frac_ones, 1 - frac_ones])

if __name__ == '__main__':
    print('CUDA' if torch.cuda.is_available() else 'CPU')

    if True:
        X = np.load('dataset/TrainDataNpz/spectrograms.npz')
        y = np.load('dataset/TrainDataNpz/labels.npz')

        X = np.array(X['a']).reshape((10000,56,2049))
        y = np.array(y['a']).reshape((10000,56))
        
        
    else:
        #load jacobs 1000 dataset
        # Load X and y
        X = np.load('dataset/spectrograms.npy')
        y = np.load('dataset/labels.npy')
    # Assert that X and y have the same number of samples
    assert X.shape[0] == y.shape[0]

    # Define train/val split
    num_samples = X.shape[0]
    train_ratio = .2
    split_index = int(num_samples*train_ratio)

    # Split X and y
    X_train = X[:split_index]
    X_val = X[split_index:]
    y_train = y[:split_index]
    y_val = y[split_index:]
    # Assert that no samples are lost
    assert X_train.shape[0] + X_val.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_val.shape[0] == y.shape[0]

    EPOCHS = 5
    BATCH_SIZE = 10
    model = ResNetModel(56,BATCH_SIZE) #(10,3,68,10)

    #make false to train from new model
    if False:
        model  = joblib.load('lstm_torch_train0.15485312044620514_val0.1472010463476181.joblib') 
        EPOCHS = 1
    
    # criterion = nn.CrossEntropyLoss(weight=distribution(y_train))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    batch_dataloader = DataLoader(Dataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(Dataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    best_validation_loss = 1
    train_loss_history = []
    val_loss_history = []
    for epoch in range(EPOCHS):
        print()
        print(f'Epoch {epoch+1}/{EPOCHS}')

        model.train()
        
        for X, y in batch_dataloader:
            
            y_p = model(X)
            loss = criterion(y_p, y)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss_history.append(loss.item())
        print("\n",zeros_and_ones(y_p[0]),"%")
        print("Accuracy",calc_acc(y, y_p).item())
        print("Loss",loss.item())
        print("Training Precision",precision_score(y.flatten().detach(),y_p.flatten().detach().round()))
        print("Training Recall",recall_score(y.flatten().detach(),y_p.flatten().detach().round()))
        print("Training f1",f1_score(y.flatten().detach(),y_p.flatten().detach().round()))
                 
        print(f'Train Loss: {np.mean(train_loss_history)}')
        print(class_counts((y_p > 0.5).int()))
        
        model.eval()
        
        for X, y in val_dataloader:
            with torch.no_grad():
                y_p = model(X)
                loss = criterion(y_p, y)
        val_loss_history.append(loss.item())
        print("\nAccuracy",calc_acc(y, y_p).item())
        print("Loss",loss.item())
        print("Validation Precision",precision_score(y.flatten().detach(),y_p.flatten().detach().round()))
        print("Validation Recall",recall_score(y.flatten().detach(),y_p.flatten().detach().round()))
        print("Validation f1",f1_score(y.flatten().detach(),y_p.flatten().detach().round()))

        

        print(f'Val Loss: {np.mean(val_loss_history)}')
        if np.mean(val_loss_history) < best_validation_loss:
            best_validation_loss = np.mean(val_loss_history)
            joblib.dump(model, f'lstm_torch_train{train_loss_history[-1]}_val{val_loss_history[-1]}.joblib')
        #training_loop = joblib.load('mlpd.joblib') 
    plot_cm(y.flatten().detach(), y_p.flatten().detach().round())
    plt.figure()
    plt.title('Loss curve')
    plt.plot(range(len(train_loss_history)), train_loss_history, label='train loss')
    plt.plot(range(len(val_loss_history)), val_loss_history, label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    