from tqdm import tqdm
import torch
from torch import nn
import torchmetrics as tm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def model_output_to_classes(model_output:torch.Tensor) -> torch.Tensor:
    return torch.max(model_output, 1)[1] # Indices of max values

def labels_to_classes(labels):
    class_labels =[]
    for c in range(int(list(labels[0].shape)[0])):
        class_labels.append([float(labels[i][c]) for i in range(len(labels))])
    return class_labels

def evaluate(model,dataloader,plot=True):
    model_output = []
    for (X, y) in tqdm(dataloader):
        model.eval()
        with torch.no_grad():
            y_p = model(X)
        model_output.extend(y_p)
    classes = labels_to_classes(model_output)

    if plot:
        plt.figure()
        plt.title('Evaluation - Labels')
        for i,c in enumerate(classes):
            plt.plot(range(len(c)), c, '.',label=str(i) +' Labels')
        plt.xlabel('Time')
        plt.ylabel('Label')
        plt.legend()
        plt.show()

    return classes

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path_to_saved_model, new_model):
    new_model.load_state_dict(torch.load(path_to_saved_model))
    new_model.eval()
    return new_model

def stats(model:nn.Module, dataloader:torch.utils.data.DataLoader, num_classes) -> float:
    model.to('cpu')
    
    precision = tm.Precision(task="multiclass", average='macro', num_classes=num_classes)
    recall = tm.Recall(task="multiclass", average='macro', num_classes=num_classes)
    f_one = tm.F1Score(task="multiclass", num_classes=num_classes)

    precisions = []
    recalls = []
    f_ones = []
    for (X, y) in tqdm(dataloader):
        X = X.to('cpu')
        y = y.to('cpu')
        model.eval()
        with torch.no_grad():
            y_p = model_output_to_classes(model(X))
            recalls.append((recall(y_p,y)).item())
            precisions.append((precision(y_p,y)).item())
            f_ones.append((f_one(y_p,y)).item())
    return sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f_ones)/len(recalls)

def plot_pred_vs_actual(model, dataloader):
    Y_p = np.array([])
    Y = np.array([])
    for (X, y) in tqdm(dataloader):
        X = X.to('cpu')
        y = y.to('cpu')
        Y = np.concatenate((Y, y.numpy()))
        
        model.eval()
        with torch.no_grad():
            Y_p = np.concatenate((Y_p, model_output_to_classes(model(X)).numpy()))
            
    print(Y_p)
    print(Y)
    plt.figure()
    plt.plot(range(len(Y)), Y, label='Actual')
    plt.legend()
    plt.show()