import numpy as np
import os
import matplotlib.pyplot as plt
import jaratoolbox
from jaratoolbox import celldatabase, ephyscore, behavioranalysis, extraplots, spikesanalysis

# Load dataframe for one mouse
oneMouseDf = celldatabase.generate_cell_database_from_subjects(["feat005"])
sessionDate = '2022-02-07'
probeDepth = 3020
oneMouseDf = oneMouseDf[(oneMouseDf.date == sessionDate) & (oneMouseDf.pdepth == probeDepth)]

# Define color palette
custom_colors = np.array([
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'])

session_types = ["pureTones", "AM", "FTVOTBorders"]
binWidth = 0.01

# Loop over individual cells
for cell_index in range(0, 20):
    one_cell_info = oneMouseDf.iloc[cell_index]
    one_cell = ephyscore.Cell(one_cell_info)

    fig, axs = plt.subplots(2, 4, figsize=(20, 6))
    behavior_class = None

    # We want 4 columns: pureTones, AM, FT, VOT
    # Define session types for iteration, splitting FTVOTBorders into two
    plot_session_types = ["pureTones", "AM", "FT", "VOT"]

    for i, plot_type in enumerate(plot_session_types):
        if plot_type == "FT" or plot_type == "VOT":
            # Load FTVOTBorders session once
            try:
                ephy_data, behavior_data = one_cell.load(sessiontype='FTVOTBorders', behavClass=behavior_class)
            except IndexError:
                print(f'Skipping cell {cell_index}, session "FTVOTBorders" not found.')
                continue

            spike_times = ephy_data['spikeTimes']
            event_onset_times = ephy_data['events']['stimOn']

            evoked_end = 1.5
            timeRange = np.array([-0.5, 1.5])
            full_time = np.arange(timeRange[0], timeRange[1], binWidth)

            spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial = (
                spikesanalysis.eventlocked_spiketimes(
                    spike_times,
                    event_onset_times,
                    timeRange,
                    spikeindex=False
                ))

            spikeCountMatrix = spikesanalysis.spiketimes_to_spikecounts(
                spikeTimesFromEventOnset,
                indexLimitsEachTrial,
                full_time
            )
            spikeCountMatrix = np.array(spikeCountMatrix)

            if plot_type == "FT":
                trialParam = behavior_data['targetFTpercent']
                possibleParams = np.unique(trialParam)
                trialsEachType = behavioranalysis.find_trials_each_type(trialParam, possibleParams)
                freqLabels = [f'{ft}%' for ft in possibleParams]
                ylabel = 'FT (%)'
                colorSource = possibleParams
            else:  # plot_type == "VOT"
                trialParam = behavior_data['targetVOTpercent']
                possibleParams = np.unique(trialParam)
                trialsEachType = behavioranalysis.find_trials_each_type(trialParam, possibleParams)
                freqLabels = [f'{vot}%' for vot in possibleParams]
                ylabel = 'VOT (%)'
                colorSource = possibleParams

            raster_colors = [custom_colors[j % len(custom_colors)] for j in range(len(colorSource))]

        else:
            # pureTones or AM normal loading
            try:
                ephy_data, behavior_data = one_cell.load(sessiontype=plot_type, behavClass=behavior_class)
            except IndexError:
                print(f'Skipping cell {cell_index}, session "{plot_type}" not found.')
                continue

            spike_times = ephy_data['spikeTimes']
            event_onset_times = ephy_data['events']['stimOn']

            if plot_type == "pureTones":
                evoked_end = 0.1
                timeRange = np.array([-0.1, 0.3])
                full_time = np.arange(timeRange[0], timeRange[1], binWidth)
            else:
                evoked_end = 1.5
                timeRange = np.array([-0.5, 1.5])
                full_time = np.arange(timeRange[0], timeRange[1], binWidth)

            spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial = (
                spikesanalysis.eventlocked_spiketimes(
                    spike_times,
                    event_onset_times,
                    timeRange,
                    spikeindex=False
                ))

            spikeCountMatrix = spikesanalysis.spiketimes_to_spikecounts(
                spikeTimesFromEventOnset,
                indexLimitsEachTrial,
                full_time
            )
            spikeCountMatrix = np.array(spikeCountMatrix)

            freqEachTrial = behavior_data['currentFreq']
            possibleFreq = np.unique(freqEachTrial)
            trialsEachType = behavioranalysis.find_trials_each_type(freqEachTrial, possibleFreq)

            if plot_type == "pureTones":
                freqLabels = ['{0:.0f}'.format(freq / 1000) for freq in possibleFreq]
                ylabel = 'Frequency (kHz)'
            else:
                freqLabels = ['{0:.0f}'.format(freq) for freq in possibleFreq]
                ylabel = 'AM Rate (Hz)'

            colorSource = possibleFreq
            raster_colors = [custom_colors[j % len(custom_colors)] for j in range(len(colorSource))]

        # Plot raster
        plt.subplot(2, 4, i + 1)
        extraplots.raster_plot(
            spikeTimesFromEventOnset,
            indexLimitsEachTrial,
            timeRange=timeRange,
            trialsEachCond=trialsEachType,
            colorEachCond=raster_colors,
            fillWidth=None,
            labels=freqLabels,
            rasterized=True
        )
        plt.title(f'{plot_type} - Raster Plot')
        plt.xlabel('Time from event onset (s)')
        plt.ylabel(ylabel)

        # Mark lines
        if plot_type == "pureTones":
            plt.axvline(0.0, color='y', linestyle='--')  # Stimulus onset
            plt.axvline(0.03, color='c', linestyle='--')  # End of onset period
            plt.axvline(0.1, color='r', linestyle='--')  # Start of offset period
            plt.axvline(0.13, color='r', linestyle='--')  # Stimulus offset
        else:
            plt.axvline(0.0, color='y', linestyle='--')
            plt.axvline(0.2, color='c', linestyle='--')
            plt.axvline(0.5, color='r', linestyle='--')
            plt.axvline(0.7, color='r', linestyle='--')

        # Plot PSTH
        plt.subplot(2, 4, i + 5)
        binsStartTime = np.arange(timeRange[0], timeRange[1], binWidth)
        extraplots.plot_psth(
            spikeCountMatrix / binWidth,
            smoothWinSize=6,
            binsStartTime=binsStartTime,
            trialsEachCond=trialsEachType,
            colorEachCond=raster_colors,
            linestyle=None,
            linewidth=3,
            downsamplefactor=1
        )
        plt.title(f'{plot_type} - PSTH')
        plt.xlabel('Time from event onset (s)')
        plt.ylabel('Firing Rate (spikes/s)')

        # Mark lines on PSTH
        if plot_type == "pureTones":
            plt.axvline(0.0, color='y', linestyle='--')  # Stimulus onset
            plt.axvline(0.03, color='c', linestyle='--')  # End of onset period
            plt.axvline(0.1, color='r', linestyle='--')  # Start of offset period
            plt.axvline(0.13, color='r', linestyle='--')  # Stimulus offset
        else:
            plt.axvline(0.0, color='y', linestyle='--')
            plt.axvline(0.2, color='c', linestyle='--')
            plt.axvline(0.5, color='r', linestyle='--')
            plt.axvline(0.7, color='r', linestyle='--')

    plt.tight_layout()
    plt.suptitle(f'Cell {cell_index}', fontsize=14)

    folder_path = '/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Singular Mouse Plots/raster_psth'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(f'{folder_path}/psth_raster_cell_{cell_index}.png')
    plt.show()
