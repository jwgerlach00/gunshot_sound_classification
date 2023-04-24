import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import joblib
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score
import seaborn as sns
from pprint import pprint
from torchmetrics import F1Score


class LSTMModel(nn.Module):
    def __init__(self, X_shape:Tuple[int, int, int]):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(X_shape[2], hidden_size=256, num_layers=1, dropout=0.2, bidirectional=True,
                            batch_first=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        print(h.shape)
        h = h.view(h.shape[1], -1)
        x = self.relu(self.fc1(h))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class LSTMDataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray, window_size:int=10):
        self.window_size = window_size
        self.X = np.reshape(X.copy(), (-1, X.shape[-1]))
        self.y = np.reshape(y.copy(), (-1, 1))
    
    def __len__(self):
        return self.X.shape[0] - self.window_size
    
    def __getitem__(self, idx:int):
        return (
            torch.tensor(self.X[idx:idx+self.window_size, :]),
            torch.tensor(self.y[idx+self.window_size]).to(torch.float32)
        )
        
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
    X = np.load('dataset/spectrograms.npz')
    y = np.load('dataset/labels.npz')
    X = np.array(X['a']).reshape((10000,56,2049))
    y = np.array(y['a']).reshape((10000,56))
    # Assert that X and y have the same number of samples
    assert X.shape[0] == y.shape[0]

    # Define train/val split
    num_samples = X.shape[0]
    train_ratio = .1
    split_index = int(num_samples*train_ratio)

    # Split X and y
    X_train = X[:split_index]
    X_val = X[split_index:2*split_index]
    y_train = y[:split_index]
    y_val = y[split_index:2*split_index]
    # Assert that no samples are lost
    # assert X_train.shape[0] + X_val.shape[0] == X.shape[0]
    # assert y_train.shape[0] + y_val.shape[0] == y.shape[0]

    model = LSTMModel(X_train.shape)

    EPOCHS = 5
    BATCH_SIZE = 10
    # criterion = nn.CrossEntropyLoss(weight=distribution(y_train))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    batch_dataloader = DataLoader(LSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(LSTMDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    f1_score = F1Score('binary', threshold=0.5, average='macro')
    
    best_validation_loss = 1
    
    def history():
        return {
            'train': {
                'loss': [],
                'acc': [],
                'f1': [],
                'precision': [],
                'recall': []
            },
            'val': {
                'loss': [],
                'acc': [],
                'f1': [],
                'precision': [],
                'recall': []
            }
        }
    
    epoch_history = history()
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        batch_history = history()

        model.train()
        for X, y in tqdm(batch_dataloader):
            
            y_p = model(X)
            print(y_p.shape,y.shape)
            loss = criterion(y_p, y)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print(f1_score(y_p, y))
            
            batch_history['train']['loss'].append(loss.item())
            batch_history['train']['acc'].append(calc_acc(y, y_p).item())
            # batch_history['train']['f1'].append(f1_score(y.flatten().detach(),y_p.flatten().detach().round()))
            # batch_history['train']['precision'].append(
            #     precision_score(y.flatten().detach(),y_p.flatten().detach().round()))
            # batch_history['train']['recall'].append(recall_score(y.flatten().detach(),y_p.flatten().detach().round()))
            
        epoch_history['train']['loss'].append(np.mean(batch_history['train']['loss']))
        epoch_history['train']['acc'].append(np.mean(batch_history['train']['acc']))
        # epoch_history['train']['f1'].append(np.mean(batch_history['train']['f1']))
        # epoch_history['train']['precision'].append(np.mean(batch_history['train']['precision']))
        # epoch_history['train']['recall'].append(np.mean(batch_history['train']['recall']))

        
        model.eval()
        for X, y in tqdm(val_dataloader):
            with torch.no_grad():
                y_p = model(X)
                loss = criterion(y_p, y)
                
            batch_history['val']['loss'].append(loss.item())
            batch_history['val']['acc'].append(calc_acc(y, y_p).item())
            # batch_history['val']['f1'].append(f1_score(y.flatten().detach(),y_p.flatten().detach().round()))
            # batch_history['val']['precision'].append(
            #     precision_score(y.flatten().detach(),y_p.flatten().detach().round()))
            # batch_history['val']['recall'].append(recall_score(y.flatten().detach(),y_p.flatten().detach().round()))
            
        epoch_history['val']['loss'].append(np.mean(batch_history['val']['loss']))
        epoch_history['val']['acc'].append(np.mean(batch_history['val']['acc']))
        # epoch_history['val']['f1'].append(np.mean(batch_history['val']['f1']))
        # epoch_history['val']['precision'].append(np.mean(batch_history['val']['precision']))
        # epoch_history['val']['recall'].append(np.mean(batch_history['val']['recall']))
        
        pprint(epoch_history)
        plot_cm(y.flatten().detach(), y_p.flatten().detach().round())

        if np.mean(epoch_history['val']['loss']) < best_validation_loss:
            best_validation_loss = np.mean(epoch_history['val']['loss'])
            joblib.dump(model, f'lstm_torch_train.joblib')

    plt.figure()
    plt.title('Loss curve')
    plt.plot(range(len(epoch_history['train']['loss'])), epoch_history['train']['loss'], label='train loss')
    plt.plot(range(len(epoch_history['val']['loss'])), epoch_history['val']['loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
