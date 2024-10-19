import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import studyparams
from jaratoolbox import celldatabase, settings, spikesanalysis, ephyscore, behavioranalysis, extraplots
from scipy import stats, signal

# %% Constants
figSavePath = "/Users/zoetomlinson/Desktop/NeuroAI/Figures/"
fontSizeLabels = 10
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
subject_list = ['feat004', 'feat005']  # , 'feat006', 'feat007', 'feat008', 'feat009', 'feat010']
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

# %% Initialize data arrays
X_pureTones = []
X_AM = []
X_speech = []
Y_brain_area_speech = []
Y_brain_area_PT = []
Y_brain_area_AM = []
Y_frequency_pureTones = []
Y_frequency_AM = []
Y_frequency_speech = []
X_array = []
Y_brain_area_array = []
Y_frequency_array = []

# %% Load dataframe
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
fullDb["recordingSiteName"] = simpleSiteNames

# %% Initialize plot and subset dataframe
for subject in subject_list:
    for date in recordingDate_list[subject]:
        for i, targetSiteName in enumerate(targetSiteNames):
            # Filter data by target site and date
            celldb = fullDb[(fullDb.subject == subject)]
            celldbSubset = celldb[(celldb.date == date)]
            celldbSubset = celldbSubset[(celldbSubset.recordingSiteName == targetSiteName)]

            # Check if the subset is empty
            if celldbSubset.empty:
                print(f"No data in {targetSiteName} on {date} for Speech, AM, and PT.")
                continue

            # Load speech behavior data for mouse
            ensembleSpeech = ephyscore.CellEnsemble(celldbSubset)
            ephysDataSpeech, bdataSpeech = ensembleSpeech.load("FTVOTBorders")
            eventOnsetTimesSpeech = ephysDataSpeech['events']['stimOn']

            spikeTimesFromEventOnsetAll, trialIndexForEachSpikeAll, indexLimitsEachTrialAll = \
                ensembleSpeech.eventlocked_spiketimes(eventOnsetTimesSpeech, timeRangeSpeech)

            FTParamsEachTrial = bdataSpeech['targetFTpercent']
            possibleFTParams = np.unique(FTParamsEachTrial)
            nTrials = len(FTParamsEachTrial)
            Y_frequency_speech.extend(FTParamsEachTrial)

            VOTParamsEachTrial = bdataSpeech['targetVOTpercent']
            possibleVOTParams = np.unique(VOTParamsEachTrial)

            trialsEachCond = behavioranalysis.find_trials_each_combination(VOTParamsEachTrial, possibleVOTParams,
                                                                           FTParamsEachTrial,
                                                                           possibleFTParams)

            spikeCounts = ensembleSpeech.spiketimes_to_spikecounts(binEdges)  # (nCells, nTrials, nBins)
            nCells = spikeCounts.shape[0]

            sumEvokedFR = spikeCounts.sum(axis=2)  # Sum across the bins so now dims are (nCells, nTrials)
            spikesPerSecEvoked = sumEvokedFR / (
                    allPeriodsSpeech[1][1] - allPeriodsSpeech[1][0])  # Divide by time width to get per sec

            trialMeans = spikesPerSecEvoked.mean(axis=1)

            spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans
            spikesPerSecEvokedNormalized = spikesPerSecEvokedNormalized.T  # (nCells, nTrials)

            if spikesPerSecEvokedNormalized.shape[1] > leastCellsArea:
                subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1],
                                               leastCellsArea, replace=False)
                spikeRateSpeechNormalized = spikesPerSecEvokedNormalized[:, subsetIndex]
            else:
                spikeRateSpeechNormalized = spikesPerSecEvokedNormalized

            nTrials = min(spikeRateSpeechNormalized.shape[1], max_trials['speech'])
            X_speech.append(spikeRateSpeechNormalized[:, :nTrials])  # (n_neurons, t_trials) for speech
            Y_brain_area_speech.extend([targetSiteName] * spikeRateSpeechNormalized.shape[0])

            # Load AM behavior data for the mouse
            ensembleAM = ephyscore.CellEnsemble(celldbSubset)
            ephysDataAM, bdataAM = ensembleAM.load("AM")
            spikeTimesAM = ephysDataAM['spikeTimes']
            nTrials = len(bdataAM['currentFreq'])
            eventOnsetTimesAM = ephysDataAM['events']['stimOn'][:nTrials]

            spikeTimesFromEventOnsetAM, trialIndexForEachSpikeAM, indexLimitsEachTrialAM = \
                ensembleAM.eventlocked_spiketimes(eventOnsetTimesAM, [evoked_start, evoked_end])

            spikeCountMatAM = ensembleAM.spiketimes_to_spikecounts(binEdges)
            spikeCountMatAM = spikeCountMatAM.sum(axis=2)
            spike_rateAM = spikeCountMatAM / (evoked_end - evoked_start)
            trialMeans = spike_rateAM.mean(axis=1)

            spike_rateAMnormalized = spike_rateAM.T - trialMeans
            spike_rateAMnormalized = spike_rateAMnormalized.T  # (nCells, nTrials)

            if spike_rateAMnormalized.shape[1] > leastCellsArea:
                subsetIndex = np.random.choice(spike_rateAMnormalized.shape[1],
                                               leastCellsArea, replace=False)
                spike_rateAMnormalized = spike_rateAMnormalized[:, subsetIndex]
            else:
                spike_rateAMnormalized = spike_rateAMnormalized

            nTrials = min(spike_rateAMnormalized.shape[1], max_trials['AM'])
            X_AM.append(spike_rateAM[:, :nTrials])  # (n_neurons, t_trials) for AM
            Y_brain_area_AM.extend([targetSiteName] * spike_rateAM.shape[0])
            Y_frequency_AM.extend([bdataAM['currentFreq'][i] for i in range(nTrials)])

            # Load Pure Tone data for the mouse
            ensemblePT = ephyscore.CellEnsemble(celldbSubset)
            ephysDataPT, bdataPT = ensemblePT.load("pureTones")
            spikeTimesPT = ephysDataPT['spikeTimes']
            nTrials = len(bdataPT['currentFreq'])
            eventOnsetTimesPT = ephysDataPT['events']['stimOn'][:nTrials]

            spikeTimesFromEventOnsetPT, trialIndexForEachSpikePT, indexLimitsEachTrialPT = \
                ensemblePT.eventlocked_spiketimes(eventOnsetTimesPT, [evoked_start, evoked_end])

            spikeCountMatPT = ensemblePT.spiketimes_to_spikecounts(binEdges)
            spikeCountMatPT = spikeCountMatPT.sum(axis=2)
            spike_ratePT = spikeCountMatPT / (evoked_end - evoked_start)
            trialMeans = spike_ratePT.mean(axis=1)

            spike_ratePTnormalized = spike_ratePT.T - trialMeans
            spike_ratePTnormalized = spike_ratePTnormalized.T  # (nCells, nTrials)

            if spike_ratePTnormalized.shape[1] > leastCellsArea:
                subsetIndex = np.random.choice(spike_ratePTnormalized.shape[1],
                                               leastCellsArea, replace=False)
                spike_ratePTnormalized = spike_ratePTnormalized[:, subsetIndex]
            else:
                spike_ratePTnormalized = spike_ratePTnormalized

            nTrials = min(spike_ratePTnormalized.shape[1], max_trials['PT'])
            X_pureTones.append(spike_ratePT[:, :nTrials])  # (n_neurons, t_trials) for PT
            Y_brain_area_PT.extend([targetSiteName] * spike_ratePT.shape[0])
            Y_frequency_pureTones.extend([bdataPT['currentFreq'][i] for i in range(nTrials)])


# %%
def adjust_array_length(X_list, max_length):
    adjusted_X_list = []
    for x in X_list:
        if x.shape[1] > max_length:
            x = x[:, :max_length]  # Truncate to max_length
        elif x.shape[1] < max_length:
            padding = np.zeros((x.shape[0], max_length - x.shape[1]))  # Pad with zeros
            x = np.hstack((x, padding))
        adjusted_X_list.append(x)
    return adjusted_X_list


# Apply adjustments
X_pureTones_adjusted = adjust_array_length(X_pureTones, max_trials['PT'])
X_AM_adjusted = adjust_array_length(X_AM, max_trials['AM'])
X_speech_adjusted = adjust_array_length(X_speech, max_trials['speech'])

# Concatenate arrays
X_speech_array = np.concatenate(X_speech_adjusted, axis=0)
X_AM_array = np.concatenate(X_AM_adjusted, axis=0)
X_PT_array = np.concatenate(X_pureTones_adjusted, axis=0)
data_dict = {}

# Create dict
for i, brain_area in enumerate(targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        if sound_type == 'speech':
            brain_area_array = np.array(Y_brain_area_speech)
            data_dict[(brain_area, sound_type)] = X_speech_array[brain_area_array == brain_area]
        if sound_type == 'AM':
            brain_area_array = np.array(Y_brain_area_AM)
            data_dict[(brain_area, sound_type)] = X_AM_array[brain_area_array == brain_area]
        if sound_type == 'PT':
            brain_area_array = np.array(Y_brain_area_PT)
            data_dict[(brain_area, sound_type)] = X_PT_array[brain_area_array == brain_area]

# Create a 3x3 grid for subplots
fig, axes = plt.subplots(3, 3, figsize=(22, 16))
fig.suptitle('Scree Plots for Different Brain Areas and Sound Types', fontsize=16)
fig.subplots_adjust(hspace=0.4, wspace=0.4)


def calculate_participation_ratio(explained_variance_ratio):
    return ((np.sum(explained_variance_ratio)) ** 2) / np.sum(explained_variance_ratio ** 2)


def plot_scree_plot(ax, data, title, y_max, particRatio):
    pca = PCA()
    pca.fit(data)
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
    ax.text(0.87, 0.85, f"Participation Ratio = {particRatio:.3f}",
            horizontalalignment='center', verticalalignment='center',
            fontsize=9, transform=ax.transAxes)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


'''
def get_max_y(data_dict):  # Determine the maximum y value across all data to ensure consistent y-axis limits
    max_y = 0
    for key, data in data_dict.items():
        pca = PCA()
        pca.fit(data)
        explained_variance_ratio = pca.explained_variance_ratio_
        max_y = max(max_y, max(explained_variance_ratio))
    return max_y * 1.1  # Add 10% padding


# Get consistent y-axis maximum value
y_max = get_max_y(data_dict)

# Plot Scree plots for each combination
for i, brain_area in enumerate(targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = data_dict[(brain_area, sound_type)]
        title = f'{brain_area} - {sound_type}, n = {data.shape[0]}'

        # Perform PCA and calculate participation ratio
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)

        pca = PCA()
        pca.fit(data_standardized)
        explained_variance_ratio = pca.explained_variance_ratio_
        particRatio = calculate_participation_ratio(explained_variance_ratio)

        # Plot the scree plot
        plot_scree_plot(axes[i, j], data, title, y_max, particRatio)

extraplots.save_figure(f"scree_pca_plots", 'png', [15, 13],
                       "/Users/zoetomlinson/Desktop/NeuroAI/Figures/Population Scree Plots")
plt.show()


def plot_pca_2d(ax, data, title):
    """
    Plot 2D PCA results on the provided axis.
    """
    pca = PCA()
    pca_result = pca.fit_transform(data)

    ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, color='black')
    ax.set_xlabel('PC 1', fontsize=12)
    ax.set_ylabel('PC 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


# Create a 3x3 grid for PCA 2D plots
fig_pca_2d, axes_pca_2d = plt.subplots(3, 3, figsize=(22, 16))
fig_pca_2d.suptitle('2D PCA Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_pca_2d.subplots_adjust(hspace=0.4, wspace=0.4)

# Plot PCA 2D results for each combination
for i, brain_area in enumerate(targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = data_dict[(brain_area, sound_type)]
        title = f'{brain_area} - {sound_type}, n = {data.shape[0]}'

        # Plot the 2D PCA
        plot_pca_2d(axes_pca_2d[i, j], data, title)

extraplots.save_figure(f"2D_pca_plots", 'png', [15, 13],
                       "/Users/zoetomlinson/Desktop/NeuroAI/Figures/Population Scree Plots")
plt.show()
'''
