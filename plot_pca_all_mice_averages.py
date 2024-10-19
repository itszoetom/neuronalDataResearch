# Plots 2D PCA and Scree Plots for all mice. Averaged trials for each frequency.
# 3x3 plots with each brain area - sound type combination

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from copy import deepcopy
import studyparams
from jaratoolbox import celldatabase, settings, spikesanalysis, ephyscore, behavioranalysis, extraplots
from scipy import stats, signal

# %% Constants
figSavePath = "/Users/zoetomlinson/Desktop/NeuroAI/Figures/"
fontSizeLabels = 10
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
subject_list = ['feat004', 'feat005', 'feat006', 'feat007', 'feat008', 'feat009', 'feat010']
recordingDate_list = {
    'feat004': ['2022-01-11', '2022-01-19', '2022-01-21'],
    'feat005': ['2022-02-07', '2022-02-08', '2022-02-11', '2022-02-14', '2022-02-15', '2022-02-16'],
    'feat006': ['2022-02-21', '2022-02-22', '2022-02-24', '2022-02-25', '2022-02-26', '2022-02-28', '2022-03-01',
                '2022-03-02'],
    'feat007': ['2022-03-10', '2022-03-11', '2022-03-15', '2022-03-16', '2022-03-18', '2022-03-21'],
    'feat008': ['2022-03-23', '2022-03-24', '2022-03-25'],
    'feat009': ['2022-06-04', '2022-06-05', '2022-06-06', '2022-06-07', '2022-06-09', '2022-06-10'],
    'feat010': ['2022-06-21', '2022-06-22', '2022-06-27', '2022-06-28', '2022-06-30']
}
targetSiteNames = ["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]
leastCellsArea = 10000
evoked_start = 0.015
evoked_end = 0.3
binWidth = 0.01
binEdges = np.arange(evoked_start, evoked_end, binWidth)
periodsNameSpeech = ['base200', 'respOnset', 'respSustained']
allPeriodsSpeech = [[-0.2, 0], [0, 0.12], [0.12, 0.24]]
timeRangeSpeech = [allPeriodsSpeech[0][0], allPeriodsSpeech[-1][-1]]
binSize = 0.005
binEdgesSpeech = np.arange(allPeriodsSpeech[1][0], allPeriodsSpeech[1][1], binSize)
max_trials = {'PT': 640, 'AM': 220, 'speech': 500}
sound_type_load = ["FTVOTBorders", "AM", "pureTones"]
previous_frequency_speech = None
previous_frequency_AM = None
previous_frequency_PT = None

# %% Load dataframe
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
fullDb["recordingSiteName"] = simpleSiteNames

# %% Initialize Data Arrays
X_speech_all = []
Y_brain_area_speech_all = []
Y_frequency_speech_all = []

X_AM_all = []
Y_brain_area_AM_all = []
Y_frequency_AM_all = []

X_pureTones_all = []
Y_brain_area_PT_all = []
Y_frequency_pureTones_all = []


# %% Initialize plot and subset dataframe
def load_data(subject, date, targetSiteName, sound_type_load):
    celldb = fullDb[(fullDb.subject == subject)]
    celldbSubset = celldb[(celldb.date == date)]
    celldbSubset = celldbSubset[(celldbSubset.recordingSiteName == targetSiteName)]

    if celldbSubset.empty:
        print(f"No data in {targetSiteName} on {date} for Speech, AM, and PT.")
        return None, None, None

    ensemble = ephyscore.CellEnsemble(celldbSubset)
    ephysData, bdata = ensemble.load(sound_type_load)

    return ensemble, ephysData, bdata


def spike_rate(sound_type, ensemble, ephysData, bdata, targetSiteName):
    X_array = []

    if sound_type == "speech":
        eventOnsetTimes = ephysData['events']['stimOn']
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, timeRangeSpeech)
        spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)
        sumEvokedFR = spikeCounts.sum(axis=2)
        spikesPerSecEvoked = sumEvokedFR / (allPeriodsSpeech[1][1] - allPeriodsSpeech[1][0])

        FTParamsEachTrial = bdata['targetFTpercent']
        VOTParamsEachTrial = bdata['targetVOTpercent']
        nTrials = len(bdata['targetFTpercent'])
        # possibleFTParams = np.unique(FTParamsEachTrial)
        # possibleVOTParams = np.unique(VOTParamsEachTrial)

        # Create and sort Y_frequency for speech
        Y_frequency = np.array([(FTParamsEachTrial[i], VOTParamsEachTrial[i]) for i in range(nTrials)])

    elif sound_type in ["AM", "PT"]:
        nTrials = len(bdata['currentFreq'])
        eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, [evoked_start, evoked_end])
        spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)
        sumEvokedFR = spikeCounts.sum(axis=2)
        spikesPerSecEvoked = sumEvokedFR / (evoked_end - evoked_start)

        # Create and sort Y_frequency for AM/PT
        Y_frequency = np.array(bdata['currentFreq'])

    trialMeans = spikesPerSecEvoked.mean(axis=1)
    spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans
    spikesPerSecEvokedNormalized = spikesPerSecEvokedNormalized.T

    if spikesPerSecEvokedNormalized.shape[1] > leastCellsArea:
        subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1], leastCellsArea, replace=False)
        spikeRateNormalized = spikesPerSecEvokedNormalized[:, subsetIndex]
    else:
        spikeRateNormalized = spikesPerSecEvokedNormalized

    X_array.append(spikeRateNormalized)
    Y_brain_area_array = [targetSiteName] * spikeRateNormalized.shape[0]
    Y_frequency_array = [Y_frequency]

    return X_array, Y_brain_area_array, Y_frequency_array


def adjust_array_and_labels(x_list, y_list, brain_area, max_length, subject, date, targetSiteName):
    adjusted_x_list = []
    adjusted_y_list = []
    adjusted_ba_list = []
    ignored_x_lists = []
    ignored_y_lists = []
    ignored_ba_lists = []

    for i, x in enumerate(x_list):
        if any(arr.shape[0] < max_length for arr in x):
            ignored_x_lists.append(f"X list {i} (lengths: {[arr.shape[0] for arr in x]})")
            ignored_ba_lists.extend(f"X list {i} (lengths: {[arr.shape[0] for arr in x]}")
            print(f"Not enough trials recorded for subject {subject}, on {date} in brain area {targetSiteName}.")
            continue

        # Truncate each array in the list to max_length
        adjusted_x_list.append([arr[:max_length] for arr in x])
        adjusted_ba_list.extend(brain_area)

    for i, y in enumerate(y_list):
        if len(y) < max_length:
            ignored_y_lists.append(f"Y list {i} (length: {len(y)})")
            ignored_ba_lists.append(f"X list {i} (lengths: {[arr.shape[0] for arr in x]}")
            continue

        adjusted_y_list.append(y[:max_length])

    return adjusted_x_list, adjusted_y_list, adjusted_ba_list, ignored_x_lists, ignored_y_lists, ignored_ba_lists


for subject in subject_list:
    for date in recordingDate_list[subject]:
        for targetSiteName in targetSiteNames:
            # Load and process data for Speech
            speechEnsemble, speechEphys, speechBdata = load_data(subject, date, targetSiteName, "FTVOTBorders")
            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = spike_rate(
                    "speech", speechEnsemble, speechEphys, speechBdata, targetSiteName)

                # Apply adjustments
                X_speech_adjusted, Y_frequency_speech_adjusted, Yba_sp_adj, ignored_x_sp, ignored_y_sp, ignored_yba_sp = (
                    adjust_array_and_labels(X_speech, Y_frequency_speech, Y_brain_area_speech, max_trials['speech'],
                                            subject, date, targetSiteName))

                if len(X_speech_adjusted) != 0:
                    # Sort Y_frequency_speech_adjusted
                    if isinstance(Y_frequency_speech_adjusted, list):
                        Y_frequency_speech_adjusted = np.array(Y_frequency_speech_adjusted[0])
                    sorted_array = Y_frequency_speech_adjusted[  # Sort by first tuple and then second
                        np.lexsort((Y_frequency_speech_adjusted[:, 1], Y_frequency_speech_adjusted[:, 0]))]

                    Y_frequency_speech_sorted = [sorted_array]
                    Y_frequency_speech_sorted = np.array(Y_frequency_speech_sorted[0])
                    indices_speech = np.argsort(Y_frequency_speech_sorted, axis=0)  # Sort by frequency combinations

                    # Check if frequency lists are all the same
                    #if previous_frequency_speech is not None:
                        #assert deepcopy(Y_frequency_speech_sorted) == previous_frequency_speech, (
                            #f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
                    #previous_frequency_speech = deepcopy(Y_frequency_speech_sorted)

                # Append to lists
                X_speech_all.extend(X_speech_adjusted)
                Y_brain_area_speech_all.extend(Yba_sp_adj)

            # Load and process data for AM
            amEnsemble, amEphys, amBdata = load_data(subject, date, targetSiteName, "AM")
            if amEnsemble:
                X_AM, Y_brain_area_AM, Y_frequency_AM = spike_rate(
                    "AM", amEnsemble, amEphys, amBdata, targetSiteName)

                # Apply adjustments
                X_AM_adjusted, Y_frequency_AM_adjusted, Yba_AM_adj, ignored_x_AM, ignored_y_AM, ignored_yba_AM = (
                    adjust_array_and_labels(X_AM, Y_frequency_AM, Y_brain_area_AM, max_trials['AM'], subject, date,
                                            targetSiteName))

                if len(X_AM_adjusted) != 0:
                    # Sort Y_frequency_AM_adjusted
                    Y_frequency_AM = np.array(Y_frequency_AM_adjusted)
                    sorted_indices = np.argsort(Y_frequency_AM)
                    sorted_Y_freq = Y_frequency_AM[0][sorted_indices]
                    Y_frequency_AM_sorted = sorted_Y_freq

                    Y_frequency_AM_sorted = np.array(Y_frequency_AM_sorted[0])
                    indices_AM = np.argsort(Y_frequency_AM_sorted)  # Sort by frequency values

                    # Check if frequency lists are all the same
                    if previous_frequency_AM is not None:
                        assert np.array_equal(Y_frequency_AM_sorted, previous_frequency_AM), (
                            f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
                    previous_frequency_AM = deepcopy(Y_frequency_AM_sorted)

                # Append to lists
                X_AM_all.extend(X_AM_adjusted)
                Y_brain_area_AM_all.extend(Yba_AM_adj)

            # Load and process data for Pure Tones
            ptEnsemble, ptEphys, ptBdata = load_data(subject, date, targetSiteName, "pureTones")
            if ptEnsemble:
                X_pureTones, Y_brain_area_PT, Y_frequency_pureTones = spike_rate(
                    "PT", ptEnsemble, ptEphys, ptBdata, targetSiteName)

                # Apply adjustments
                X_PT_adjusted, Y_frequency_PT_adjusted, Yba_PT_adj, ignored_x_PT, ignored_y_PT, ignored_yba_PT = (
                    adjust_array_and_labels(X_pureTones, Y_frequency_pureTones, Y_brain_area_PT, max_trials['PT'],
                                            subject, date, targetSiteName))

                if len(X_PT_adjusted) != 0:
                    # Convert Y_frequency_pureTones_adjusted
                    Y_frequency_pureTones = np.array(Y_frequency_PT_adjusted)
                    sorted_indices = np.argsort(Y_frequency_pureTones)
                    sorted_Y_freq = Y_frequency_pureTones[0][sorted_indices]
                    Y_frequency_pureTones_sorted = sorted_Y_freq

                    Y_frequency_pureTones_sorted = np.array(Y_frequency_pureTones_sorted[0])
                    indices_PT = np.argsort(Y_frequency_pureTones_sorted)  # Sort by frequency values

                    # Check if frequency lists are all the same
                    if previous_frequency_PT is not None:
                        assert np.array_equal(Y_frequency_pureTones_sorted, previous_frequency_PT), (
                            f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
                    previous_frequency_PT = deepcopy(Y_frequency_pureTones_sorted)

                # Append to the lists
                X_pureTones_all.extend(X_PT_adjusted)
                Y_brain_area_PT_all.extend(Yba_PT_adj)


def sort_x_arrays(X_list, indices):
    sorted_x_list = []
    for x in X_list:
        sorted_x = [arr[indices] for arr in x]
        sorted_x_list.append(np.array(sorted_x))
    return sorted_x_list


# Apply sorting to the X arrays
X_AM_sorted = sort_x_arrays(X_AM_all, indices_AM)
X_PT_sorted = sort_x_arrays(X_pureTones_all, indices_PT)
X_speech_sorted = sort_x_arrays(X_speech_all, indices_speech)

# Concatenate the sorted arrays
X_speech_array = np.concatenate(X_speech_sorted, axis=0)
X_speech_array = X_speech_array[:, :, 0]  # flatten to change shape from (neurons, 500, 2) to (neurons, 500)

X_AM_array = np.concatenate(X_AM_sorted, axis=0)
X_PT_array = np.concatenate(X_PT_sorted, axis=0)

# Initialize dictionary to hold average spike rates per frequency
data_dict = {}
y_max = 0.8
avg_spike_rate_dict = {}
X_mean = []

# Add data to the dictionary for each brain area and sound type
for brain_area in targetSiteNames:
    # For speech
    brain_area_array_speech = np.array(Y_brain_area_speech_all)
    X_speech_array_adjusted = X_speech_array[brain_area_array_speech == brain_area]
    X_speech_array_adjusted = X_speech_array_adjusted.T
    Y_frequency_speech_array_adjusted = Y_frequency_speech_sorted
    data_dict[(brain_area, 'speech')] = {'X': X_speech_array_adjusted, 'Y': Y_frequency_speech_array_adjusted}

    unique_speech_freqs = np.unique(Y_frequency_speech_array_adjusted, axis=0)
    X_means = []

    for freq in unique_speech_freqs:
        # Create a mask by comparing each element of Y_frequency_speech_array_adjusted to the tuple freq
        freq_mask = np.all(Y_frequency_speech_array_adjusted == freq, axis=1)
        # Apply the mask to X_speech_array_adjusted
        freq_values = X_speech_array_adjusted[freq_mask]
        # Calculate the mean spike rate for this frequency
        if freq_values.size > 0:
            col_means = np.mean(freq_values, axis=0)
            X_means.append(col_means)

    X_means = np.vstack(X_means)

    avg_spike_rate_dict[(brain_area, 'speech')] = {'X': np.array(X_means), 'Y': unique_speech_freqs}

    # For AM
    brain_area_array_AM = np.array(Y_brain_area_AM_all)
    X_AM_array_adjusted = X_AM_array[brain_area_array_AM == brain_area]
    X_AM_array_adjusted = X_AM_array_adjusted.T
    Y_frequency_AM_array_adjusted = Y_frequency_AM_sorted
    data_dict[(brain_area, 'AM')] = {'X': X_AM_array_adjusted, 'Y': Y_frequency_AM_array_adjusted}

    unique_am_freqs = np.unique(Y_frequency_AM_array_adjusted)
    X_means = []

    for freq in unique_am_freqs:
        # Get the indices of all trials with the current frequency
        freq_mask = (Y_frequency_AM_array_adjusted == freq)
        # Apply mask to the trials and calculate the mean spike rate for this frequency
        freq_values = X_AM_array_adjusted[freq_mask]
        col_means = np.mean(freq_values, axis=0)
        col_means = col_means.T
        X_means.append(col_means)

    X_means = np.vstack(X_means)

    avg_spike_rate_dict[(brain_area, 'AM')] = {'X': np.array(X_means), 'Y': unique_am_freqs}

    # For pure tones
    brain_area_array_PT = np.array(Y_brain_area_PT_all)
    X_PT_array_adjusted = X_PT_array[brain_area_array_PT == brain_area]
    X_PT_array_adjusted = X_PT_array_adjusted.T
    Y_frequency_PT_array_adjusted = Y_frequency_pureTones_sorted
    data_dict[(brain_area, 'PT')] = {'X': X_PT_array_adjusted, 'Y': Y_frequency_PT_array_adjusted}

    unique_pt_freqs = np.unique(Y_frequency_PT_array_adjusted)
    X_means = []

    for freq in unique_pt_freqs:
        # Get the indices of all trials with the current frequency
        freq_mask = (Y_frequency_PT_array_adjusted == freq)
        # Apply mask to the trials and calculate the mean spike rate for this frequency
        freq_values = X_PT_array_adjusted[freq_mask, :]
        col_means = np.mean(freq_values, axis=0)
        col_means = col_means.T
        X_means.append(col_means)

    X_means = np.vstack(X_means)

    avg_spike_rate_dict[(brain_area, 'PT')] = {'X': X_means, 'Y': unique_pt_freqs}

# Create a 3x3 grid for subplots
fig_scree, axes_scree = plt.subplots(3, 3, figsize=(22, 16))
fig_scree.suptitle('Scree Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_scree.subplots_adjust(hspace=0.4, wspace=0.4)


def calculate_participation_ratio(explained_variance_ratio):
    return ((np.sum(explained_variance_ratio)) ** 2) / np.sum(explained_variance_ratio ** 2)


def plot_scree_plot(ax, data, title, y_max, particRatio):
    pca = PCA()
    pca.fit(data['X'])
    explained_variance_ratio = pca.explained_variance_ratio_

    n_components = len(explained_variance_ratio)
    x_max = min(n_components, 13)  # Limit x-axis to 13 components
    x_min = 0

    ax.bar(range(x_max), explained_variance_ratio[:x_max], color='black')
    ax.set_xlabel('PCA features', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(x_max))
    ax.set_xlim(x_min, 13)  # Set x-axis limits from 0 to 13
    ax.set_ylim(0, y_max)  # Set y-axis limits to be consistent
    ax.text(0.6, 0.85, f"Participation Ratio = {particRatio:.3f}",
            horizontalalignment='center', verticalalignment='center',
            fontsize=14, transform=ax.transAxes)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


# Plot Scree plots for each combination
for i, brain_area in enumerate(targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = avg_spike_rate_dict.get((brain_area, sound_type), None)
        data_full = data_dict.get((brain_area, sound_type))
        if data is None:
            continue  # Skip if no data available for this combination

        title = f'{brain_area} - {sound_type} n = {data_full["X"].shape[1]}'

        # Perform PCA and calculate participation ratio
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data['X'])

        pca = PCA()
        pca.fit(data['X'])
        explained_variance_ratio = pca.explained_variance_ratio_
        particRatio = calculate_participation_ratio(explained_variance_ratio)

        # Plot the scree plot
        plot_scree_plot(axes_scree[i, j], data, title, y_max, particRatio)

# Save Scree plots figure
fig_scree.show()
fig_scree.savefig("/Users/zoetomlinson/Desktop/NeuroAI/Figures/Population Plots/PopScreeAveragePlots.png")

# Create a 3x3 grid for 2D PCA subplots
fig_pca, axes_pca = plt.subplots(3, 3, figsize=(22, 16))
fig_pca.suptitle('2D PCA Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_pca.subplots_adjust(hspace=0.4, wspace=0.4)


# Function to create a 2D PCA plot with color-coded points based on frequency
def plot_2d_pca(ax, data, labels, title, cmap='viridis'):
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data['X'])

    scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap=cmap, s=32)
    ax.set_title(title)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    plt.colorbar(scatter, ax=ax, orientation='vertical')


# Plot 2D PCA plots
for i, brain_area in enumerate(targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = avg_spike_rate_dict.get((brain_area, sound_type), None)
        data_full = data_dict.get((brain_area, sound_type))
        if data is None:
            continue  # Skip if no data is available

        title = f'{brain_area} - {sound_type}, n = {data_full["X"].shape[1]}'

        # For 'speech' sound type, create a mapping of frequencies to numbers
        if sound_type == 'speech':
            Y_labels = [tuple(row) for row in data["Y"]]
            unique_labels = [(0, 0), (0, 33), (0, 67), (0, 100), (33, 100), (67, 100), (100, 100), (100, 67), (100, 33),
                             (100, 0), (67, 0), (33, 0)]
            label_to_number = {label: idx for idx, label in enumerate(unique_labels)}
            color_values = np.array([label_to_number[label] for label in Y_labels])
            plot_2d_pca(axes_pca[i, j], data, color_values, title)

        # For 'AM' sound type, directly use the 'Y' values
        elif sound_type == 'AM':
            Y_labels = np.array(data_dict[(brain_area, sound_type)]['Y'])
            plot_2d_pca(axes_pca[i, j], data, unique_am_freqs, title)

        # For 'PT' sound type, apply log10 transformation to 'Y'
        elif sound_type == 'PT':
            Y_labels = np.array(data_dict[(brain_area, sound_type)]['Y'])
            plot_2d_pca(axes_pca[i, j], data, unique_pt_freqs, title)

# Save as pngs
fig_pca.savefig("/Users/zoetomlinson/Desktop/NeuroAI/Figures/Population Plots/2D_PCA_Average_Plots.png")
plt.show()