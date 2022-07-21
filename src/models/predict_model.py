import argparse
import json
import pandas as pd
from src.models.utils import load_dataset
import tensorflow as tf

# python -m src.models.predict_model 

def save_results(config, test_loss, test_accuracy):
    cfg = config["train"]

    column_names = ["experiment_desc", "data_type", "window_length", "epochs", "batch_size", "dropout", "learning_rate", "trait", "test_loss", "test_accuracy"]
    df = pd.DataFrame([[cfg["experiment_desc"], config["dataset"]["save_suffix"][1:], config["dataset"]["window_length"],
                             cfg["epochs"], cfg["batch_size"], cfg["dropout"], cfg["learning_rate"], cfg["trait"], test_loss, test_accuracy]], columns=column_names)

    df.to_csv(cfg["save_path"] + "predictions//" + cfg["experiment_desc"] + ".csv", index=False)


def predict(config):

    cfg = config["train"]

    # load test
    _, _, _, _, X_test, y_test = load_dataset(config)

    # load best model
    exp_desc = cfg["experiment_desc"]
    best_model = tf.keras.models.load_model(cfg["model_save_path"] + exp_desc + '-best.h5')
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=2)
    print(f"Experiment : {exp_desc}")
    print("Best model")
    print("Test accuracy: {:5.2f}%".format(100 * test_accuracy))
    print("Test loss: {:5.4f}".format(test_loss))

    save_results(config, test_loss, test_accuracy)

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

            predict(config)