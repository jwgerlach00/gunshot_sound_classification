import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import joblib
from matplotlib import pyplot as plt
from tqdm import tqdm


class LSTMModel(nn.Module):
    def __init__(self, X_shape:Tuple[int, int, int]):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(X_shape[2], hidden_size=256, num_layers=1, dropout=0.2, bidirectional=False,
                            batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)
        x = self.relu(self.fc1(h))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class LSTMDataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray, window_size:int=10):
        self.window_size = window_size
        self.X = np.reshape(X.copy(), (-1, X.shape[-1]))
        self.y = np.reshape(y.copy(), (-1, 1))
        # self.y = y.copy()
    
    def __len__(self):
        return self.X.shape[0] - self.window_size
    
    def __getitem__(self, idx:int):
        return (
            torch.tensor(self.X[idx:idx+self.window_size, :]),
            torch.tensor(self.y[idx+self.window_size]).to(torch.float32)
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

    EPOCHS = 100
    BATCH_SIZE = 32
    # criterion = nn.CrossEntropyLoss(weight=distribution(y_train))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    batch_dataloader = DataLoader(LSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(LSTMDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    best_validation_loss = 1
    for epoch in range(EPOCHS):
        print()
        print(f'Epoch {epoch+1}/{EPOCHS}')

        model.train()
        train_loss_history = []
        train_acc_history = []
        for X, y in tqdm(batch_dataloader):
            
            y_p = model(X)
            loss = criterion(y_p, y)
            # print("\n",zeros_and_ones(y_p[0]),"%")
            # print("Accuracy",calc_acc(y, y_p).item())
            # print("Loss",loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_history.append(loss.item())
            train_acc_history.append(calc_acc(y, y_p).item())
            
        print(f'Train Loss: {np.mean(train_loss_history)}')
        print(f'Train Accuracy: {np.mean(train_acc_history)}')
        
        model.eval()
        val_loss_history = []
        val_acc_history = []
        for X, y in tqdm(val_dataloader):
            with torch.no_grad():
                y_p = model(X)
                loss = criterion(y_p, y)
                val_loss_history.append(loss.item())
                val_acc_history.append(calc_acc(y, y_p).item())
               
        print(f'Val Loss: {np.mean(val_loss_history)}')
        print(f'Val Accuracy: {np.mean(val_acc_history)}')
        # if np.mean(val_loss_history) < best_validation_loss:
        #     best_validation_loss = np.mean(val_loss_history)
        #     joblib.dump(model, f'lstm_torch_train{train_loss_history[-1]}_val{val_loss_history[-1]}.joblib')
        #training_loop = joblib.load('mlpd.joblib') 

    # plt.figure()
    # plt.title('Loss curve')
    # plt.plot(range(len(train_loss_history)), train_loss_history, label='train loss')
    # plt.plot(range(len(val_loss_history)), val_loss_history, label='val loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()