import argparse
import json
import numpy as np
import pandas as pd
import sklearn.metrics
from src.models.utils import load_dataset
import tensorflow as tf

# python -m src.models.kfold_predict_model 

def save_results(config, fold, test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa):
    cfg = config["train"]

    if fold == 1:
        column_names = ["experiment_desc", "fold", "data_type", "window_length", "epochs", "batch_size", "dropout", "learning_rate", "trait",
                    "test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_kappa"]
        df = pd.DataFrame([[cfg["experiment_desc"], fold, config["dataset"]["save_suffix"][1:], config["dataset"]["window_length"],
                             cfg["epochs"], cfg["batch_size"], cfg["dropout"], cfg["learning_rate"], cfg["trait"],
                             test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa]], columns=column_names)
    elif fold != 1:
        df = pd.read_csv(cfg["save_path"] + "predictions//" + cfg["experiment_desc"] + "_kfold" + ".csv")
        column_names = df.columns
        df_new = pd.DataFrame([[cfg["experiment_desc"], fold, config["dataset"]["save_suffix"][1:], config["dataset"]["window_length"],
                             cfg["epochs"], cfg["batch_size"], cfg["dropout"], cfg["learning_rate"], cfg["trait"],
                             test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa]], columns=column_names)
        df = pd.concat([df, df_new])
    
    df.to_csv(cfg["save_path"] + "predictions//" + cfg["experiment_desc"] + "_kfold" + ".csv", index=False)


def kfold_predict(config):

    cfg = config["train"]

    # load test
    _, _, _, _, X_test, y_test = load_dataset(config)

    accuracy = 0
    loss = 0
    precision = 0
    recall = 0
    f1 = 0
    kappa = 0

    num_folds = cfg["k_folds"]
    for fold in np.arange(1, num_folds+1):

        # load best model
        model_name = cfg["experiment_desc"] + "_" + 'fold-' + str(fold) 

        best_model = tf.keras.models.load_model(cfg["model_save_path"] + model_name + '-best.h5')
        test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=2)
        y_pred = best_model.predict(X_test)
        test_precision, test_recall, test_f1, _ = sklearn.metrics.precision_recall_fscore_support(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), pos_label=1, average='binary')
        test_kappa = sklearn.metrics.cohen_kappa_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        print(f"Experiment : {model_name}")
        print("Fold: {:d}".format(fold))
        print("Test accuracy: {:5.2f}%".format(100 * test_accuracy))
        print("Test loss: {:5.4f}".format(test_loss))
        print("Test precision: {:5.4f}".format(test_precision))
        print("Test recall: {:5.4f}".format(test_recall))
        print("Test F1: {:5.4f}".format(test_f1))
        print("Test Kappa: {:f}".format(test_kappa))

        # average metrics over the 5 folds
        accuracy = accuracy + test_accuracy
        loss = loss + test_loss
        precision = precision + test_precision
        recall = recall + test_recall
        f1 = f1 + test_f1
        kappa = kappa + test_kappa

        save_results(config, fold, test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa)

    save_results(config, "avg", loss/num_folds, accuracy/num_folds, precision/num_folds, recall/num_folds, f1/num_folds, kappa/num_folds)

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

            kfold_predict(config)