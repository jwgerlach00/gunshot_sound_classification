import numpy as np
import torch
from torch.utils.data import DataLoader
from lstm_torch import LSTMModel, LSTMDataset


X = np.load('../../../ece542_repos/gunshot_sound_classification/dataset/spectrograms.npy')
y = np.load('../../../ece542_repos/gunshot_sound_classification/dataset/labels.npy')
model = LSTMModel(X.shape)

model.load_state_dict(torch.load('model_lstm_test.torch'))

dataloader = DataLoader(LSTMDataset(X, y), batch_size=1, shuffle=False)

print(X.shape)

print(model(torch.tensor(X[1, :, :])))
exit()

model.eval()
with torch.no_grad():
    for x, y in dataloader:
        print(x.shape)
        break
        print(model(x))
        print(y)
        break