import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import joblib
from matplotlib import pyplot as plt


class ResNetModel(nn.Module):
    def __init__(self,x_shape, BATCH_SIZE):
        super(ResNetModel, self).__init__()
        self.batch = BATCH_SIZE
        self.input = nn.Linear(1000, 512)
        self.relu = nn.ReLU() # Activation function
        self.length = x_shape[-2]
        self.hidden_layer = nn.Linear(512, x_shape[-2])
        self.output = nn.Sigmoid()
        self.rn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
    def forward(self, x):
        x = x.reshape((self.batch,3,self.length,43))
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

    EPOCHS = 100
    BATCH_SIZE = 10
    model = ResNetModel(X_train.shape,BATCH_SIZE)

    
    # criterion = nn.CrossEntropyLoss(weight=distribution(y_train))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    batch_dataloader = DataLoader(Dataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(Dataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    best_validation_loss = 1
    for epoch in range(EPOCHS):
        print()
        print(f'Epoch {epoch+1}/{EPOCHS}')

        model.train()
        train_loss_history = []
        for X, y in batch_dataloader:
            
            y_p = model(X)
            loss = criterion(y_p, y)
            print("\n",zeros_and_ones(y_p[0]),"%")
            print("Accuracy",calc_acc(y, y_p).item())
            print("Loss",loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_history.append(loss.item())
            
        print(f'Train Loss: {np.mean(train_loss_history)}')
        print(class_counts((y_p > 0.5).int()))
        
        model.eval()
        val_loss_history = []
        for X, y in val_dataloader:
            with torch.no_grad():
                y_p = model(X)
                loss = criterion(y_p, y)
                val_loss_history.append(loss.item())
               
        print(f'Val Loss: {np.mean(val_loss_history)}')
        if np.mean(val_loss_history) < best_validation_loss:
            best_validation_loss = np.mean(val_loss_history)
            joblib.dump(model, f'lstm_torch_train{train_loss_history[-1]}_val{val_loss_history[-1]}.joblib')
        #training_loop = joblib.load('mlpd.joblib') 

    plt.figure()
    plt.title('Loss curve')
    plt.plot(range(len(train_loss_history)), train_loss_history, label='train loss')
    plt.plot(range(len(val_loss_history)), val_loss_history, label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()