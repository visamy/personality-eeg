import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
from src.models.models import EEGNet
from src.models.utils import load_dataset, plot_loss_accuracy
import tensorflow as tf

# python -m src.models.train_model --c "experiments//exp01.json"

def train(config):

    cfg = config["train"]

    # load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(config)

    # compile model
    model = EEGNet(nb_classes = 2, Chans = 14, Samples = config["dataset"]["window_length"]*config["dataset"]["sampling_frequency"], dropoutRate = cfg["dropout"], 
                    kernLength = int(128/2), dropoutType = cfg["dropout_type"], F1 = cfg["temporal_filters"], 
                    D = cfg["spatial_filters_depth"], F2 = cfg["temporal_filters"]*cfg["spatial_filters_depth"])

    optimizer = tf.keras.optimizers.Adam(learning_rate = cfg["learning_rate"])
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = cfg["optimizer_metric"])

    # callbacks
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) # tensorboard cb
    path_model = cfg["model_save_path"] + cfg["experiment_desc"] + '-best.h5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(path_model, verbose=1, monitor=cfg["model_save_metric"], save_best_only=True, mode="auto") # model checkpoints
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg["early_stop_epochs"], verbose=1) # early stopping
    path_logs = cfg["save_path"] + "logs//LOGS_" + cfg["experiment_desc"] + '.csv'
    log_csv = tf.keras.callbacks.CSVLogger(path_logs, separator = ',', append=False) # csv logger

    if cfg["save_best_model"] == True:
        callback_list = [checkpointer, early_stop, log_csv] #, tensorboard_callback]
    else:
        callback_list = [early_stop, log_csv]

    # train
    hist = model.fit(x = X_train, y = y_train, batch_size = cfg["batch_size"], epochs = cfg["epochs"], verbose = 1, validation_data = (X_val, y_val), callbacks = callback_list)

    # save model
    model.save(cfg["model_save_path"] + cfg["experiment_desc"] + '-' + str(cfg["epochs"]) + 'epochs_' + str(cfg["batch_size"]) + 'batch_' + str(cfg["dropout"]) + 'dropout_' + str(cfg["learning_rate"]) + 'lr.h5')

    # print scores
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Results...')
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # plot
    fold = 0
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

            train(config)