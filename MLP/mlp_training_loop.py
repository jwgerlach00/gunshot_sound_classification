from helpers.preprocessing import get_model_params
from mlp_architecture import MLPModel, MLPDataset
from helpers import TrainingLoop
import joblib
from torchsummary import summary
import torch

if __name__ == '__main__':
    hyperparams = get_model_params()

    train = True #set to train and save, or load and eval
    reload = False

    if train == True:
        if reload == True:
            training_loop = joblib.load('mlpd.joblib') 
        else:
            training_loop = TrainingLoop(MLPModel, MLPDataset, hyperparams, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #summary(training_loop.model,(32,100))
        training_loop.training_loop(old_data=False)
        joblib.dump(training_loop, 'mlpd.joblib') #save model
    else:
        training_loop = joblib.load('mlpd.joblib') 

    model_accuracy = training_loop.accuracy(training_loop.model,training_loop.dataloader(MLPDataset, hyperparams, hyperparams['validation_amount'], torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    print('Model accuracy: {0}%\n'.format(model_accuracy))
    training_loop.plot_loss()