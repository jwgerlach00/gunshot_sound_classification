import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class LSTMModel(nn.Module):
    def __init__(self, X_shape:Tuple[int, int, int]):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(X_shape[2], hidden_size=256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, X_shape[1])
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)
        x = self.fc1(h)
        x = self.relu(x)
        x = self.fc2(x)
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

if __name__ == '__main__':
    print('CUDA' if torch.cuda.is_available() else 'CPU')

    # Load X and y
    X = np.load('dataset/spectrograms.npy')
    y = np.load('dataset/labels.npy')
    # Assert that X and y have the same number of samples
    assert X.shape[0] == y.shape[0]

    # Define train/val split
    num_samples = X.shape[0]
    train_ratio = .8
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

    EPOCHS = 10
    BATCH_SIZE = 10
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        batch_dataloader = DataLoader(LSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
        
        model.train()
        lost_history = []
        for X, y in batch_dataloader:
            
            y_p = model(X)
            loss = criterion(y_p, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lost_history.append(loss.item())
            
        print(f'Loss: {np.mean(lost_history)}')
            
            