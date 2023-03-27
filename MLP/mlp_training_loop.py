from helpers.preprocessing import get_imu_data,get_model_params
from mlp_architecture import MLPModel, MLPDataset
from helpers import TrainingLoop
import joblib

if __name__ == '__main__':
    hyperparams = get_model_params()
    imu, ann = get_imu_data() #gets data

    train = True #set to train and save, or load and eval
    if train == True:
        training_loop = TrainingLoop(MLPModel, MLPDataset, hyperparams)
        training_loop.training_loop(imu, ann)
        joblib.dump(training_loop, 'mlp.joblib') #save model
    else:
        training_loop = joblib.load('mlp.joblib') 

    print('Train accuracy: {0}%\nVal accuracy: {1}%'.format(training_loop.train_acc, training_loop.val_acc))
    training_loop.plot_loss()