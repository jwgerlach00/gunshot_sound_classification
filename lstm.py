import tensorflow as tf
from tensorflow import keras
from helpers import AudioSampler, WindowGenerator
from typing import Tuple
import numpy as np
import tensorflow_addons as tfa


f1_score = tfa.metrics.F1Score(
    num_classes = 2,
    average = 'macro', # Macro to handle class imbalance
    name = 'f1_score'
)

def create_model(X_shape:Tuple[int, int, int]):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_shape[1], X_shape[2])))
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=256, 
                recurrent_dropout = .2
            )
        )
    )
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(X_shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


if __name__ == '__main__':
    # Show GPU output
    print(tf.config.list_physical_devices('GPU'))
    
    # Load X and y
    X = np.load('dataset/spectrograms.npy')
    y = np.load('dataset/labels.npy')
    # Assert that X and y have the same number of samples
    assert X.shape[0] == y.shape[0]
    
    # Define train/val split
    num_samples = X.shape[0]
    train_ratio = .8
    split_index = int(num_samples*train_ratio)
    
    # Split X and y
    X_train = X[:split_index]
    X_val = X[split_index:]
    y_train = y[:split_index]
    y_val = y[split_index:]
    # Assert that no samples are lost
    assert X_train.shape[0] + X_val.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_val.shape[0] == y.shape[0]
    
    # Instantiate model
    model = create_model(X_train.shape)
    print(model.summary())
    
    # Hyperparams
    BATCH_SIZE = 10
    EPOCHS = 20
    
    # Define callbacks
    # Create a callback that saves the model's weights every 5 epochs
    checkpoint_path = 'training_checkpoints/cp-best-f1_score.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        monitor = 'val_acc',
        mode = 'max',
        save_best_only=True,
    )
    
    # Fit model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        shuffle=False,
        callbacks=[cp_callback],
        # class_weight=class_weight
    )
    print(model.predict(X_train).shape)
    
    # model.save('saved_model/lstm')
