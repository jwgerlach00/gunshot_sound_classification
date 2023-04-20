from torch import nn
import torch
from helpers.AudioSampler import AudioSampler

class MLPModel(nn.Module):
    def __init__(self, hyperparams:dict):
        super(MLPModel, self).__init__()
        self.input = nn.Linear(hyperparams['window_size'], hyperparams['hidden_size'])
        self.relu = nn.ReLU() # Activation function
        self.hidden_layer = nn.Linear(hyperparams['hidden_size'], hyperparams['num_classes'])
        self.output = nn.Tanh()
        
    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.hidden_layer(x)
        return self.output(x)


class MLPDataset(torch.utils.data.Dataset):
    data = AudioSampler()

    def __init__(self, size, device, hyperparams):
        self.device = device
        self.window_size = hyperparams['window_size']
        self.X, self.y = MLPDataset.data.sample_array(size, self.window_size, convert_to_mono=True,output_spectrogram=True)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return torch.tensor(self.X[idx], device=self.device), torch.tensor(self.y[idx], device=self.device)
