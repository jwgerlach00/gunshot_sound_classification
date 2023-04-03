import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers import eval
from mlp_architecture import MLPDataset, MLPModel
import yaml
from torch.utils.data import DataLoader
from helpers import TrainingLoop


with open('MLP/mlp_hyperparams.yaml', 'r') as f:
    hyperparams = yaml.safe_load(f)
model = eval.load_model('f2_model.torch', MLPModel(hyperparams))

eval_generator =  TrainingLoop.dataloader(MLPDataset, hyperparams, 100, torch.device('cpu'))
# out = eval.stats(model, eval_generator, num_classes=2)
# print(out)

eval.plot_pred_vs_actual(model, eval_generator)
    
