from torch import nn
import torch

class MLPModel(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, hyperparams:dict):
        super(MLPModel, self).__init__()
        hidden_layer_sizes = hyperparams['mlp']['hidden_layers']
        
        self.relu = nn.ReLU() # Activation function
        self.input = nn.Linear(6, hidden_layer_sizes[0])

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            if i < len(hidden_layer_sizes) - 1:
                self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1])\
                    .to(device=MLPModel.device))
            else:
                self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i], hyperparams['num_classes'])\
                    .to(device=MLPModel.device))
        # self.hidden_layer1 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        # self.hidden_layer2 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        # self.hidden_layer3 = nn.Linear(hidden_layer_sizes[2], hyperparams['num_classes'])
        
        # self.hidden_layers = []
        # self.hidden_layers.append(self.hidden_layer1)
        # self.hidden_layers.append(self.hidden_layer2)
        # self.hidden_layers.append(self.hidden_layer3)
        
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.input(x)
        # x = self.relu(x)
        # x = self.hidden_layer1(x)
        # x = self.relu(x)
        # x = self.hidden_layer2(x)
        # x = self.relu(x)
        # x = self.hidden_layer3(x)
        # print(self.hidden_layer3)
        
        for hidden_layer in self.hidden_layers:
            x = self.relu(x)
            x = hidden_layer(x) # No relu at end because output is softmax

        return self.output(x)


class MLPDataset(torch.utils.data.Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, X, y, *args, **kwargs):
        self.X = torch.tensor(X, device=MLPDataset.device).float()
        self.y = torch.tensor(y, device=MLPDataset.device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
