# Personality traits classification from EEG signals using EEGNet
This repository contains the source code for reproducing results in the paper ["Personality traits classification from EEG signals using EEGNet"](https://ieeexplore.ieee.org/document/9843118) by V. Guleva, A. Calcagno, P. Reali and A. M. Bianchi.

# Quickstart
## Setup
1. Clone the repository:
```
git clone https://github.com/visamy/personality-eeg
```
```
cd personality-eeg
```
2. Create a new virtual environment:
```
python -m venv env
```
3. Activate environment:
  * Windows
```
source env/Scripts/activate
```
  * MacOS/Linux
```
source env/bin/activate
```
4. Install requirements:
```
pip install -r requirements.txt
```
5. Install `src`:
```
pip install -e .
```

## Download Dataset
The required files from the [AMIGOS dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/index.html) are `physiological recordings: Original data in Matlab format` and `metadata: Microsoft Excel (.xlsx) spreadsheets`, available for download [here](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html).

The .mat files need be stored in `/data/raw/` and the "Participants_Personality.xlsx" spreadsheet file need be stored in `/data/`.

## EEGNet
The model used for training is EEGNet as implemented in [`https://github.com/vlawhern/arl-eegmodels`](https://github.com/vlawhern/arl-eegmodels).

# Reproduce results
All the needed configurations are set in the `config/config.json` file, to be modified as needed.

Six illustrative Jupyter notebooks are provided in `notebooks/`

## 1. Dataset Creation
Run the script
```
python -m src.data.make_dataset --config "config//config.json"
```

To reproduce the three types of dataset used for the paper, edit the `"dataset"` object in `config.json` as specified in the paper.

## 2. (Optional) Hyperparameter tuning
This step is not required for reproducing the results. 

To tune the hyperparameters of the model, run the script
```
python -m src.hyperparameters.tune_hyperparameters --tuner "hyperband"
```
The `--tuner` can be set to `"hyperband"`, `"bayesian"`, or `"random"` search algorithms. The hyperparameter tuning was implemented using [KerasTuner](https://github.com/keras-team/keras-tuner).

The hyperparameters and tuning configuration can be edited in the `"hyperparameter_tuning"` object in `config.json`.

To obtain the tuning results, refer to the Jupyter notebook `notebooks/3-hyperparameter-tuning-results.ipynb`

## 2. Training
For the k-fold training, run the script
```
python -m src.models.kfold_train_model 
```

Note that training is separate for each personality trait. To obtain results for all five traits, edit the `"train"` object in `config.json` by changing the `"trait"` element. The element `"experiment_desc"` controls the naming of all the subsequent reports.

Alternatively, a simple 1-fold training is also implemented and can be run with the script:
```
python -m src.models.train_model --config "experiments//exp01.json"
```
Other config files can be used for experimental results. An `exp01.json` config example file is provided in `experiments/` and can be used as above.

## 3. Predictions
To predict the results of the k-fold trained models, run the script
```
python -m src.models.kfold_predict_model
```
Loss, Accuracy, Precision, Recall, F1, and Kappa metrics are calculated on the test set for each k-fold and on average over all folds. The results are saved in a `.csv` file in `reports/predictions`.

To predict the results of the 1-fold trained models, run the script
```
python -m src.models.predict_model
```

## 4. Visualizations

### Filters and hidden layer outputs
The filters and hidden layer outputs can be visualized as shown in the `notebooks/5-visualizations-filters-and-outputs.ipynb` Jupyter notebook. 

### Attributions
The attributions of the inputs are obtained as implemented in the [DeepExplain](https://github.com/marcoancona/DeepExplain) framework, using the DeepLIFT algorithm.

To obtain the attributions for the test inputs, run the script
```
python -m src.visualizations.attributions --model_path "models//trained_model_name.h5"
```
The path to the targeted trained model has to be specified.

The calculated attributions are saved in `models/attributions`, while the generated figures from the attributions are saved in `reports/attributions/`.
