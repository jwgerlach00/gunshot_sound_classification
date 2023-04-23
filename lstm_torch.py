import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class LSTMModel(nn.Module):
    def __init__(self, X_shape:Tuple[int, int, int]):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(X_shape[2], hidden_size=256, batch_first=True)
        self.fc1 = nn.Linear(256, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, X_shape[1])
        
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)
        x = self.relu(self.fc1(h))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        x = self.sigmoid(x)
        return x


class LSTMDataset(Dataset):
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

if __name__ == '__main__':
    print('CUDA' if torch.cuda.is_available() else 'CPU')
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class LSTMModel(nn.Module):
    def __init__(self, X_shape:Tuple[int, int, int]):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(X_shape[2], hidden_size=512, batch_first=True)
        # self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, X_shape[1])
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)
        # x = self.fc1(h)
        # x = self.relu(x)
        x = self.fc2(h)
        x = self.sigmoid(x)
        return x


class LSTMDataset(Dataset):
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
        
def calc_acc(y_p, y):
    return torch.sum((y_p > 0.5).int() == y) / (y.shape[0] * y.shape[1])

def class_counts(y):
    return [torch.sum(y == 0), torch.sum(y == 1)]

def distribution(y):
    frac_ones = np.sum(y) / (y.shape[0] * y.shape[1])
    return torch.tensor([frac_ones, 1 - frac_ones])

def zeros_and_ones(t):
    ones = torch.tensor((t*2),dtype=torch.long).sum().detach().item()
    total = len(t.detach().numpy())
    return (total-ones)/total*100,ones/total*100

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

    model = LSTMModel(X_train.shape)

    EPOCHS = 1000
    BATCH_SIZE = 10
    # criterion = nn.CrossEntropyLoss(weight=distribution(y_train))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    batch_dataloader = DataLoader(LSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(LSTMDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        model.train()
        train_loss_history = []
        for X, y in batch_dataloader:
            
            y_p = model(X)
            loss = criterion(y_p, y)
            print(zeros_and_ones(y_p[0]),"%")
            print(calc_acc(y_p, y))
            
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
