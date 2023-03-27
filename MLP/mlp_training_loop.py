from helpers.preprocessing import read_all_data
import yaml
from mlp_architecture import MLPModel, MLPDataset
from helpers import TrainingLoop
import joblib
import torch

if __name__ == '__main__':
    data_dict = read_all_data()
    imu = data_dict['imu'].to_numpy()
    ann = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    with open('MLP/mlp_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
        
    training_loop = TrainingLoop(MLPModel, MLPDataset, hyperparams)
    # training_loop.training_loop(imu, ann)
    # joblib.dump(training_loop, 'mlp_training_loop_yes_weight_yes_norm_2.joblib')


    # training_loop = joblib.load('mlp_training_loop.joblib')
    training_loop = torch.load('mlp_training_loop.joblib',map_location=torch.device('cpu'))

    print('Train accuracy: {0}%\nVal accuracy: {1}%'.format(training_loop.train_acc, training_loop.val_acc))
    training_loop.plot_loss()