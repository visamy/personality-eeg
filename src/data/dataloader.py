import mne
import numpy as np
from keras.utils import np_utils
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

def extract_personality_data(path):
    """
    Reads the personality data Excel file 'Participants_Personality.xlsx', returns a pandas dataframe and the missing subjects list
    """
    df = pd.read_excel(path, sheet_name='Personalities', nrows=6, header=None).T
    df.columns = df.iloc[0] 
    df = df[1:] # FORMATTING: sets first row (UserID, Extroversion) as header
    df.rename(columns = {"Creativity (openness)":"Openness"}, inplace="True")
    subjects = list(df['UserID'])
    missing_subjects = sorted(set(range(int(subjects[0]), int(subjects[-1]) + 1)).difference(subjects))
    return df, missing_subjects

def create_binary_labels(personality_df, trait, threshold):
    """
    Binarizes the personality data based on the given threshold.

    Input: 
        - personality_df: personality dataframe extracted with "extract_personality_data"
        - trait: trait to binarize
        - threshold: threshold value >= 1, <= 7
    Possible traits are = 'Extroversion', 'Agreeableness', 'Conscientiousness',
                          'Emotional Stability', 'Openness'
    """
    subjects = list(personality_df['UserID'])
    if trait not in personality_df.columns:
        raise ValueError('There is no {trait} personality trait, possible traits are = "Extroversion", "Agreeableness", "Conscientiousness", "Emotional Stability", "Openness"'.format(trait=repr(trait)))

    if threshold < 1 or threshold > 7:
        raise ValueError

    ptrait = [(0 if x < threshold else 1) for x in list(personality_df[trait])]
    global_labels = pd.DataFrame(list(zip(subjects, ptrait)), columns=['Subject', trait])
    return global_labels

def preprocess_signal(raw, info, band = [0.1,45], scale=True):
    """
    Given the EEG signal in MNE raw format and its info file, it:
        - bandpass filters (0.1-45 Hz as default)
        - scales with mean scaler (StandardScaler)
    Returns: np_array of shape (1,14,datapoints)
    """
    raw = raw.filter(l_freq=band[0], h_freq=band[1])
    filt_signal = raw.get_data() # FILTERED SIGNAL

    if scale == True:
        scaler = mne.decoding.Scaler(info, scalings='mean')
        e = filt_signal.reshape(1, filt_signal.shape[0], filt_signal.shape[1]) # fictitious epochs, reshape as (1,14,datapoints)
        filt_signal = scaler.fit_transform(e) # mean scaled signal
        
    return filt_signal

def read_eeg_mat(path, subject):
    subject_suff = str(subject).zfill(2)
    data = loadmat(path + 'Data_Original_P' + subject_suff + '.mat') # load file associated to current subject

    # trial = one EEG acquisition corresponding to one short video experiment. all trials have a 14 x nr. samples shape
    trials = [] # initialize the trails of each subject. the number of trials is usually 16 but some subjects have trails missing.
    for epo in range(16): # base epochs is 16
        if data['EEG_DATA'][0,epo].size == 0: # check if trial is missing and in case, skip it
            #print(f'Trial {epo+1} missing')
            continue
        
        trial = data['EEG_DATA'][0,epo][:,3:17].transpose()
        trials.append(trial)
    return trials

def split_trials(trials, split):
    if (split[0] + split[1] + split[2]) != 100:
        raise ValueError('Percentage splits don\'t sum up to 100.')

    num_trials = len(trials) # number of trials for the current subject

    # split files of a subject in [70%] for training, [15%] for validation, [15%] for test
    num_train = int((split[0] / 100) * num_trials)
    num_val   = round((split[1] / 100) * num_trials)
    num_test  = round((split[2] / 100) * num_trials)

    np.random.seed(737)
    np.random.shuffle(trials) # shuffle files

    train = trials[0:num_train]
    val   = trials[num_train:num_train+num_val]
    test  = trials[num_train+num_val:num_train+num_val+num_test] 
    return train, val, test

def segment_trial(trial, label, info, preprocess: bool, bandpass, window_length: int, window_overlap: int):
    """
    Segments each trial using a 3s (default) window with no overlap. If specified, pre-processes the signal using a bandpass filter.
    """
    raw = mne.io.RawArray(trial, info, verbose=0) # create Raw class signal (MNE-python)

    if preprocess == True:
        pre_processed_signal = preprocess_signal(raw, info, band=bandpass, scale=False) # PREPROCESS                
        raw = mne.io.RawArray(pre_processed_signal.squeeze(), info, verbose=0)

    windows = mne.make_fixed_length_epochs(raw, duration = window_length, overlap = window_overlap, verbose=0)
    windows = windows.get_data() # the windows obtained have shape (nr. of windows, 14, nr of samples = window_length * 128)
    labels = np.ones(shape=(windows.shape[0],))*label # the labels have shape (nr. of windows, )
    return windows, labels

def scale_data(info, eeg_train, eeg_val, eeg_test):
    """
    Scale train and validation data using Standard Scaler
    """
    eeg_train_val = np.vstack((eeg_train, eeg_val))
    epochs_train =  mne.EpochsArray(eeg_train_val, info, verbose=0)
    epochs_test =  mne.EpochsArray(eeg_test, info, verbose=0)
    scaler = mne.decoding.Scaler(info, scalings='mean')
    X = scaler.fit_transform(epochs_train.get_data())
    X_train = X[0:eeg_train.shape[0], :, :]  
    X_val = X[eeg_train.shape[0]::, :, :]
    X_test = scaler.transform(epochs_test.get_data())
    return X_train, X_val, X_test

def tf_reshape(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Reshapes the arrays for the tensorflow CNN models use
    """
    # convert data to NHWC format (N =  batch == number of windows, H = height = EEG channels, W = width = window samples, C= number of channels = kernels == 1)
    X_train = np.expand_dims(X_train, 3)
    X_val   = np.expand_dims(X_val, 3)
    X_test  = np.expand_dims(X_test, 3)

    # convert labels to categorical values
    y_train = np_utils.to_categorical(y_train, num_classes = 2)
    y_val = np_utils.to_categorical(y_val, num_classes = 2)
    y_test = np_utils.to_categorical(y_test, num_classes = 2)
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataset(global_labels, path, missing_subjects = [], split = [70,15,15], window_length = 3, window_overlap = 0, preprocess = True, bandpass = [0.1,45], scale = True):
    '''
    Reads the AMIGOS EEG .mat files and creates train, validation and test sets according to the split (default split = (70,15,15))
    It splits the data proportionally among each set for each subject:
        example: subject 1 has 16 files, with a specified split of (70 train, 15 val, 15 test), 8 files will go to the training set, 4 will go to the validation set and 4 will go to the test set.
    It also scales, preprocesses and reshapes the data if needed.

    Every file is segmented into few-seconds windows with or without overlap.

    Inputs:
        - global_labels: the trait-label associated to each subject obtained with "create_binary_labels_threshold"
        - path: raw .mat files path
        - missing_subjects: subjects that have no personality data, to be discarded
        - split: the proportional train/val/test split percentages 
        - window_length: the length of the segmentation window (default 3 seconds)
        - window_overlap: segmentation overlap (default 0)
        - preprocess: boolean, if data should be preprocessed
        - bandpass: pre-processing bandpass filter (default: 1-45 Hz)
        - scale: boolean, if data should be rescaled 
    '''

    # initialize
    X_train  = []
    y_train  = []
    X_val    = []
    y_val    = []
    X_test   = []
    y_test   = []

    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    info     = mne.create_info(ch_names, 128, ch_types='eeg').set_montage('biosemi64')

    for subject in tqdm (np.arange(1,41), desc="Subject: " ): 
        if subject not in missing_subjects:

            trials = read_eeg_mat(path, subject) # obtain trials for current subject

            train, val, test = split_trials(trials, split) # split the trials into train, validation and test sets

            label = int(global_labels.loc[global_labels['Subject'] == subject][global_labels.columns[1]]) # match the subject to its corresponding label by using the global_labels dataframe
            
            # segment all trials            
            for trial in train:
                windows, labels = segment_trial(trial, label, info, preprocess = preprocess, bandpass = bandpass, window_length = window_length, window_overlap = window_overlap)
                X_train.append(windows)
                y_train.extend(labels)

            for trial in val:
                windows, labels = segment_trial(trial, label, info, preprocess = preprocess, bandpass = bandpass, window_length = window_length, window_overlap = window_overlap)
                X_val.append(windows)
                y_val.extend(labels)

            for trial in test:
                windows, labels = segment_trial(trial, label, info, preprocess = preprocess, bandpass = bandpass, window_length = window_length, window_overlap = window_overlap)
                X_test.append(windows)
                y_test.extend(labels)
        
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_val = np.vstack(X_val)
    y_val = np.vstack(y_val)
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)

    if scale == True:
        X_train, X_val, X_test = scale_data(info, X_train, X_val, X_test)

    X_train, y_train, X_val, y_val, X_test, y_test = tf_reshape(X_train, y_train, X_val, y_val, X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
