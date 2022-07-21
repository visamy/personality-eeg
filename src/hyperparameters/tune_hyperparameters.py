import argparse
from functools import partial
import keras_tuner as kt
import json
from src.models.utils import load_dataset
from src.models.models import EEGNet
import tensorflow as tf

# python -m src.hyperparameters.tune_hyperparameters --tuner "bayesian"

def model_builder_partial(hp, config):

    cfg = config["hyperparameter_tuning"]

    # define hyperparameters search space
    hp_dropout = hp.Float('dropout', min_value=cfg["dropout"][0], max_value=cfg["dropout"][0], step=0.1)
    hp_dropout_type = hp.Choice("dropout_type", values=cfg["dropout_type"])
    hp_learning_rate = hp.Float("lr", min_value=cfg["learning_rate"][0], max_value=cfg["learning_rate"][1], sampling="log")
    hp_F1 = hp.Int('F1', min_value=cfg["temporal_filters"][0], max_value=cfg["temporal_filters"][1], step=1, sampling='log')
    hp_D = hp.Int('D', min_value=cfg["spatial_filters_depth"][0], max_value=cfg["spatial_filters_depth"][1], step=1, sampling='log')

    # compile model
    model = EEGNet(nb_classes = 2, Chans = 14, Samples = config["dataset"]["window_length"]*config["dataset"]["sampling_frequency"], dropoutRate = hp_dropout, 
                    kernLength = int(128/2), dropoutType = hp_dropout_type, F1 = hp_F1, 
                    D = hp_D, F2 = hp_F1*hp_D)

    optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = config["train"]["optimizer_metric"])
    
    return model


def define_tuner(config, tuner_name):
    cfg = config["hyperparameter_tuning"]

    model_builder = partial(model_builder_partial, config = config)

    if tuner_name == "hyperband":
        tuner = kt.Hyperband(
                        model_builder,
                        objective=cfg["tuner_objective"],
                        max_epochs = cfg["max_epochs"],
                        seed=737,
                        factor = 10,
                        hyperband_iterations=1,
                        directory=cfg["directory"],
                        project_name=config["train"]["experiment_desc"]
        )

    elif tuner_name == "bayesian":
        tuner = kt.BayesianOptimization(
                        model_builder,
                        objective=cfg["tuner_objective"],
                        num_initial_points=2,
                        alpha=0.0001,
                        beta=2.6,
                        seed=737,
                        directory=cfg["directory"],
                        project_name=config["train"]["experiment_desc"]
        )
    elif tuner_name == "random":
        tuner = kt.RandomSearch(
                        model_builder,
                        objective=cfg["tuner_objective"],
                        max_trials=10,
                        seed=737,
                        directory=cfg["directory"],
                        project_name=config["train"]["experiment_desc"]
        )

    return tuner

def tune(config, args):

    cfg = config["hyperparameter_tuning"]
    tuner_name = args.tuner

    tuner = define_tuner(config, tuner_name)

    # early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor=cfg["tuner_objective"], patience=0.1*cfg["max_epochs"])

    # load dataset
    X_train, y_train, X_val, y_val, _, _ = load_dataset(config)
    tuner.search(X_train, y_train, epochs=cfg["epochs"], validation_data=(X_val, y_val), callbacks=[stop_early])

    return
    

if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="config//config.json", help="Path to configuration file.")
    parser.add_argument("-a", "--tuner", dest="tuner", default="hyperband", help="Choose tuner algorithm: ['hyperband', 'bayesian', 'random']")
    args = parser.parse_args()

    # Ensure a config was passed to the script.
    if not args.config:
        print("No configuration file provided.")
        exit()
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

            if args.tuner not in ['hyperband', 'bayesian', 'random']:
                print("Chosen tuner algorithm not valid.")
                exit()

            tune(config, args)