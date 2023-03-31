from torch import nn
import torch
from helpers.AudioSampler import AudioSampler

class MLPModel(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, hyperparams:dict):
        super(MLPModel, self).__init__()
        hidden_layer_sizes = hyperparams['mlp']['hidden_layers']
        
        self.relu = nn.ReLU() # Activation function
        self.input = nn.Linear(hyperparams['mlp']['input_size'], hidden_layer_sizes[0])

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            if i < len(hidden_layer_sizes) - 1:
                self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1])\
                    .to(device=MLPModel.device))
            else:
                self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i], hyperparams['num_classes'])\
                    .to(device=MLPModel.device))
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.input(x)
        
        for hidden_layer in self.hidden_layers:
            x = self.relu(x)
            x = hidden_layer(x) # No relu at end because output is softmax

        return self.output(x)


class MLPDataset(torch.utils.data.Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = AudioSampler('city.wav','kaggle_sounds/Zastava M92/9 (1).wav')
    def __init__(self,hyperparams,*args, **kwargs):
        self.window_size = hyperparams['window_size']
        self.X,self.y = MLPDataset.data.sample_array(1000, convert_to_mono=True)
        print(self.X.shape,self.y.shape)
    
    def __len__(self):
        return len(self.X) - self.window_size
    
    def __getitem__(self,idx):
        self.X = torch.tensor(self.X, device=MLPDataset.device).float()
        self.y = torch.tensor(self.y, device=MLPDataset.device)
        return self.X[idx:idx+self.window_size], self.y[idx:idx+self.window_size]
