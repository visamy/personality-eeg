import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import json
import numpy as np
from deepexplain.tf.v2_x import DeepExplain
from src.models.utils import load_dataset
import tensorflow.compat.v1.keras.backend as K

import warnings
warnings.filterwarnings("ignore")

# python -m src.visualizations.attributions --model_path "models//EEGNet-8,2_Agreeableness_bpscaled-1,45Hz-best.h5"

def get_attributions(model_path):

    _, _, _, _, X_test, y_test = load_dataset(config)

    # clear tensorflow session
    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(model_path)

    with DeepExplain(session = K.get_session()) as de:
        input_tensor   = model.layers[0].input # input layer
        fModel         = tf.keras.models.Model(inputs = input_tensor, outputs = model.layers[-2].output)  
        target_tensor  = fModel(input_tensor)#input_tensor)    # change target if you want attributions for other intermediate layers and not the input

        # attributions
        deeplift_attributions = de.explain('deeplift', target_tensor * y_test, input_tensor, X_test)
    
    return deeplift_attributions


def save_attributions(deeplift_attributions, config):

    np.save(config["attributions"]["attributions_path"] + config["train"]["experiment_desc"] + "_attributions.npy", deeplift_attributions)

    return

if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="config//config.json", help="Path to configuration file.") # can load separate experiment configs from /experiments
    parser.add_argument("-m", "--model_path", dest="model_path", help="Path to trained model.")

    args = parser.parse_args()

    # Ensure a config was passed to the script.
    if not args.config:
        print("No configuration file provided.")
        exit()
    if not args.model_path:
        print("No model file provided.")
        exit()        
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

        deeplift_attributions = get_attributions(args.model_path)
        save_attributions(deeplift_attributions, config)

