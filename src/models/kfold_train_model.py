import argparse
from keras.utils import np_utils
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.models.models import EEGNet
from src.models.utils import load_dataset, plot_loss_accuracy
import tensorflow as tf


# python -m src.models.kfold_train_model 

def kfold_train(config):

    cfg = config["train"]

    # load dataset
    X_train, y_train, X_val, y_val, _, _ = load_dataset(config)
    eeg_train = np.vstack((X_train, X_val))
    labels_train = np.vstack((y_train, y_val))
    labels_train = np.argmax(labels_train, axis=1)

    folds = list(StratifiedKFold(n_splits = cfg["k_folds"], shuffle = True, random_state = 737).split(eeg_train, labels_train))

    for j, (train_idx, val_idx) in enumerate(folds):
            
        print('\nFold ', j+1)
        X_train_cv = eeg_train[train_idx]
        y_train_cv = labels_train[train_idx]
        X_val_cv = eeg_train[val_idx]
        y_val_cv= labels_train[val_idx]

        y_train_cv = np_utils.to_categorical(y_train_cv)
        y_val_cv = np_utils.to_categorical(y_val_cv)

        # compile model
        model = EEGNet(nb_classes = 2, Chans = 14, Samples = config["dataset"]["window_length"]*config["dataset"]["sampling_frequency"], dropoutRate = cfg["dropout"], 
                        kernLength = int(128/2), dropoutType = cfg["dropout_type"], F1 = cfg["temporal_filters"], 
                        D = cfg["spatial_filters_depth"], F2 = cfg["temporal_filters"]*cfg["spatial_filters_depth"])

        optimizer = tf.keras.optimizers.Adam(learning_rate = cfg["learning_rate"])
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = cfg["optimizer_metric"])

        model_name = cfg["experiment_desc"] + "_" + 'fold-' + str(j+1) 

        # callbacks
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) # tensorboard cb
        path_model = cfg["model_save_path"] + model_name + '-best.h5'
        checkpointer = tf.keras.callbacks.ModelCheckpoint(path_model, verbose=1, monitor=cfg["model_save_metric"], save_best_only=True, mode="auto") # model checkpoints
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg["early_stop_epochs"], verbose=1) # early stopping
        path_logs = cfg["save_path"] + "logs//LOGS_" + model_name + '.csv'
        log_csv = tf.keras.callbacks.CSVLogger(path_logs, separator = ',', append=False) # csv logger

        if cfg["save_best_model"] == True:
            callback_list = [checkpointer, early_stop, log_csv] #, tensorboard_callback]
        else:
            callback_list = [early_stop, log_csv]

        # train
        hist = model.fit(x = X_train_cv, y = y_train_cv, batch_size = cfg["batch_size"], epochs = cfg["epochs"], verbose = 1, validation_data = (X_val_cv, y_val_cv), callbacks = callback_list)

        # print predictions
        score = model.evaluate(X_val_cv, y_val_cv, verbose=1)
        print('Results...')
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        print()

        # PLOT
        fold = j+1
        plot_loss_accuracy(hist, config, fold)

    return


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="config//config.json", help="Path to configuration file.") # can load separate experiment configs from /experiments

    args = parser.parse_args()

    # Ensure a config was passed to the script.
    if not args.config:
        print("No configuration file provided.")
        exit()
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

            kfold_train(config)