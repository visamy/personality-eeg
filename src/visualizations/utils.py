import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

def plot_attributions(attributions, config, channels, scalings = 1e1):

    # plot style
    plt.rc('font', size=14) #controls default text size
    plt.rc('axes', titlesize=18) #fontsize of the title
    plt.rc('axes', labelsize=16) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=14) #fontsize of the y tick labels
    plt.rc('legend', fontsize=16) #fontsize of the legend

    fig = plt.figure(figsize=(10,7))
    title = 'Attributions for ' + config["train"]["experiment_desc"]

    im = plt.imshow(np.mean(attributions, axis=0).squeeze()*scalings, aspect='auto', cmap ='turbo')
    plt.title(title)
    plt.yticks(np.arange(0,14,1), channels)
    plt.xlabel('Samples')
    plt.ylabel('Channels')

    cb_ax = fig.add_axes([0.95, 0.25, 0.03, 0.5])
    cbar = fig.colorbar(im, cax=cb_ax, pad=0.8, aspect=7, shrink=0.5,extend='both')
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(1)
    cbar.outline.set_visible(True)

    return fig
        


def plot_attributions_topomap(attributions, config, channels, scalings = 1e3, title = "Attributions"):

    matplotlib.style.use('default')
    
    # info object
    sfreq = config["dataset"]["sampling_frequency"]
    info = mne.create_info(channels, sfreq, ch_types='eeg').set_montage('biosemi64')

    # Epochs and Evoked object
    epochs = mne.EpochsArray(attributions.mean(axis=2)*1000, info, verbose=0) # shape = (attr, channels, time points, 1) => average in time (dim=2)
    evoked = epochs.average()

    # mask for plot
    mask = np.zeros(evoked.data.shape, dtype='bool')
    mask.fill(True)

    # topomap
    fig = evoked.plot_topomap(times=[0], cmap='turbo', units=dict(eeg=''), scalings=scalings, title=title, colorbar=True, show_names=False, time_format="", res=300, sphere=(0, 0, 0.035, 0.094),
                        mask = mask, mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4), outlines='skirt')

    return fig
