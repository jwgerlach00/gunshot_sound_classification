from torch import nn
import torch
from helpers.AudioSampler import AudioSampler

class MLPModel(nn.Module):
    def __init__(self, hyperparams:dict):
        super(MLPModel, self).__init__()
        self.input = nn.Linear(hyperparams['window_size'], 20)
        self.relu = nn.ReLU() # Activation function
        self.hidden_layer = nn.Linear(20, hyperparams['num_classes'])
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.hidden_layer(x)
        return self.output(x)


class MLPDataset(torch.utils.data.Dataset):
    data = AudioSampler()

    def __init__(self,hyperparams,size):
        self.window_size = hyperparams['window_size']
        self.X,self.y = MLPDataset.data.sample_array(size,self.window_size, convert_to_mono=True)
        self.X = torch.tensor(self.X,requires_grad=True)
        self.y = torch.tensor(self.y,requires_grad=True)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
