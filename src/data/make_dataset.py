import argparse
import json
import numpy as np
from src.data import dataloader

# python -m src.data.make_dataset --config "config//config.json"

def make_dataset(config):
    traits = config["traits"] # 5 personality traits names 
    cfg = config["dataset"]
    save_path = cfg["save_path"] # dataset save path
    save_suffix = cfg["save_suffix"] # dataset save suffix to be added (e.g.: bandpass_nodelta_scaled)

    df, missing_subjects = dataloader.extract_personality_data(path = cfg["personality_path"])

    for trait in traits: 
        print(f"Trait {trait}")
        global_labels = dataloader.create_binary_labels(df, trait, df[trait].mean()) # create labels using mean as threshold

        X_train, y_train, X_val, y_val, X_test, y_test = dataloader.create_dataset(global_labels, cfg["eeg_path"], missing_subjects = missing_subjects,
                                                                             split = cfg["split"], window_length = cfg["window_length"], 
                                                                             window_overlap = cfg["window_overlap"], preprocess = cfg["preprocess"], 
                                                                             bandpass = cfg["bandpass"], scale = cfg["scale"])

        np.save(save_path + "eeg_train_" + traits[trait] + save_suffix + ".npy", X_train)
        np.save(save_path + "eeg_val_" + traits[trait] + save_suffix + ".npy", X_val)
        np.save(save_path + "eeg_test_" + traits[trait] + save_suffix + ".npy", X_test)

        np.save(save_path + "labels_train_" + traits[trait] + save_suffix + ".npy", y_train)
        np.save(save_path + "labels_val_" + traits[trait] + save_suffix + ".npy", y_val)
        np.save(save_path + "labels_test_" + traits[trait] + save_suffix + ".npy", y_test)
    return
    
if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="config//config.json", help="Path to configuration file.")
    args = parser.parse_args()

    # Ensure a config was passed to the script.
    if not args.config:
        print("No configuration file provided.")
        exit()
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

            make_dataset(config)