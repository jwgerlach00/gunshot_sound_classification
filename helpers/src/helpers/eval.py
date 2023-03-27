from tqdm import tqdm
import torch
from torch import nn
import torchmetrics as tm


def model_output_to_classes(model_output:torch.Tensor) -> torch.Tensor:
    return torch.max(model_output, 1)[1] # Indices of max values


def stats(model:nn.Module, dataloader:torch.utils.data.DataLoader,num_classes) -> float:
    precisions = []
    recalls = []
    f_ones = []
    precision = tm.Precision(task="multiclass", average='macro', num_classes=num_classes)
    recall = tm.Recall(task="multiclass", average='macro', num_classes=num_classes)
    f_one = tm.F1Score(task="multiclass", num_classes=num_classes)

    for (X, y) in tqdm(dataloader):
        model.eval()
        with torch.no_grad():
            y_p = model_output_to_classes(model(X))

            recalls.append((recall(y_p,y)).item())
            precisions.append((precision(y_p,y)).item())
            f_ones.append((f_one(y_p,y)).item())
    return sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f_ones)/len(recalls)
