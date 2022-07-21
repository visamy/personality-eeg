from matplotlib import pyplot as plt
import numpy as np
from src.models.models import EEGNet
import tensorflow as tf

def load_dataset(config):

    path = config["dataset"]["save_path"]
    suffix = config["dataset"]["save_suffix"]
    cfg = config["train"]

    X_train = np.load(path + "eeg_train_" + config["traits"][cfg["trait"]] + suffix + ".npy")
    X_val = np.load(path + "eeg_val_" + config["traits"][cfg["trait"]] + suffix + ".npy")
    X_test = np.load(path + "eeg_test_" + config["traits"][cfg["trait"]] + suffix + ".npy")
    
    y_train = np.load(path + "labels_train_" + config["traits"][cfg["trait"]] + suffix + ".npy")
    y_val = np.load(path + "labels_val_" + config["traits"][cfg["trait"]] + suffix + ".npy")
    y_test = np.load(path + "labels_test_" + config["traits"][cfg["trait"]] + suffix + ".npy")

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_loss_accuracy(hist, config, fold):
    
    cfg = config["train"]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 10), constrained_layout=True)
    if fold != 0:
        plt.suptitle(cfg["experiment_desc"] + " Fold-" + str(fold) + ': ' + str(cfg["batch_size"]) + ' batch size, ' + str(cfg["dropout"]) + ' dropout, ' + str(cfg["learning_rate"]) + ' learning rate, ' + "\n" + config["dataset"]["save_suffix"][1:] + ' data, ' + cfg["trait"] + ' trait' )
    else:
        plt.suptitle(cfg["experiment_desc"] + ': ' + str(cfg["batch_size"]) + ' batch size, ' + str(cfg["dropout"]) + ' dropout, ' + str(cfg["learning_rate"]) + ' learning rate, ' + "\n" + config["dataset"]["save_suffix"][1:] + ' data, ' + cfg["trait"] + ' trait' )
    ax1.plot(hist.history["loss"])
    ax1.plot(hist.history["val_loss"])
    ax1.set_title("model loss")
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")

    ax2.plot(hist.history['accuracy'])
    ax2.plot(hist.history['val_accuracy'])
    ax2.set_title('model accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')

    if cfg["save_figure"] == True:
        if fold != 0:
            fig.savefig(cfg["save_path"] + "figures//" + cfg["experiment_desc"] + "_" + 'fold-' + str(fold) + "_" + str(cfg["epochs"]) + 'epochs_' + str(cfg["batch_size"]) + 'batch_' + str(cfg["dropout"]) + 'dropout_' + str(cfg["learning_rate"]) + 'lr_' + config["traits"][cfg["trait"]] + config["dataset"]["save_suffix"] + ".png", bbox_inches="tight", dpi=150)
        else:
            fig.savefig(cfg["save_path"] + "figures//" + cfg["experiment_desc"] + "_" + str(cfg["epochs"]) + 'epochs_' + str(cfg["batch_size"]) + 'batch_' + str(cfg["dropout"]) + 'dropout_' + str(cfg["learning_rate"]) + 'lr_' + config["traits"][cfg["trait"]] + config["dataset"]["save_suffix"] + ".png", bbox_inches="tight", dpi=150)

    return