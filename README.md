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
The main model used 

DeepExplain



Startup
python -m venv env

source env/Scripts/activate

pip install -r requirements.txt
pip install -e .
