This repository contains the source code for reproducing results in the paper ["Personality traits classification from EEG signals using EEGNet"](10.1109/MELECON53508.2022.9843118).

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
The required files from the [AMIGOS dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/index.html) are `physiological recordings: Original data in Matlab format` and `metadata: Microsoft Excel (.xlsx) spreadsheets`, available for download at `[http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html)`.

The .mat files need be stored in `/data/raw` and the "Participants_Personality.xlsx" spreadsheet file need be stored in `/data/`.

## EEGNet
The model used for training is EEGNet as implemented in `https://github.com/vlawhern/arl-eegmodels`.

# Reproduce results
All the needed configuration are set in the `config/config.json` file, to be modified as needed.

## 1. Create Dataset
Run the script
```
python -m src.data.make_dataset --config "config//config.json"
```

To reproduce the three types of dataset used for the paper, edit the `"dataset"` object in `config.json` as specified in the paper.

## 2. Train
For the k-fold training, run the script
```
python -m src.models.kfold_train_model --config "config//config.json"
```

Note that training is separate for each personality trait. To obtain results for all five traits, edit the `"train"` object in `config.json` by changing the `"trait"` element.

Alternatively, a simple 1-fold training is also implemented and can be run with the script:
```
python -m src.models.train_model --c "experiments//exp01.json"
```
Other config files can be used for experimental results. An `exp01.json` example is provided in `experiments/`.

## 3. Predict
To predict the results of the k-fold trained models, run the script
```
python -m src.models.kfold_predict_model --config "config//config.json"
```
Loss, Accuracy, Precision, Recall, F1, and Kappa metrics are calculated on the test set for each k-fold and on average over all folds. The results are saved in a `.csv` file in `reports/predictions`

