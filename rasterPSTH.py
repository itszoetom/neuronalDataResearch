import numpy as np
import os
import matplotlib.pyplot as plt
import jaratoolbox
from jaratoolbox import celldatabase, ephyscore, behavioranalysis, extraplots

# Electrophysiological and behavioral neuron data. Calculating firing rate plotting Raster and PSTH Graphs. Individual neurons in one mouse on one  day. 

# Loads in dataframe for one mouse
oneMouseDf = jaratoolbox.celldatabase.generate_cell_database_from_subjects(["feat004"])
sessionDate = '2022-01-11'
probeDepth = 2318
oneMouseDf = oneMouseDf[(oneMouseDf.date == sessionDate) & (oneMouseDf.pdepth == probeDepth)]


# Selects one cell/neuron and loads it
for cell_index in range(0, 109):  # max 204
    one_cell_info = oneMouseDf.iloc[cell_index]
    one_cell = jaratoolbox.ephyscore.Cell(one_cell_info)

    session_types = ["pureTones", "AM"]
    behavior_class = None

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust the figsize as needed

    # Loads the ephys and behavior data for that cell
    for i, session_type in enumerate(session_types):
        ephy_data, behavior_data = one_cell.load(sessiontype=session_type,
                                                 behavClass=behavior_class)

        # Gets spike times and event onset times (i.e. when the sounds were presented or the stim turns on)
        spike_times = ephy_data['spikeTimes']
        event_onset_times = ephy_data['events'][
            'stimOn']

        # Sets time ranges for plotting and calculating firing rates
        baseline_start = -0.1
        baseline_end = 0.3
        evoked_start = 0.015
        evoked_end = 0.3
        binWidth = 0.01
        full_time = np.arange(baseline_start, evoked_end, binWidth)
        timeRange = np.array([baseline_start, evoked_end])

        baseline_rates = []
        evoked_rates = []

        for event_time in event_onset_times:
            # Baseline calculation
            baseline_spike_count = len(spike_times) / (baseline_end - baseline_start)
            baseline_rates.append(baseline_spike_count)

            # Evoked calculation
            evoked_spike_count = len(spike_times) / (evoked_end - evoked_start)
            evoked_rates.append(evoked_spike_count)

        # Lines up event times and spike times
        spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial = jaratoolbox.spikesanalysis.eventlocked_spiketimes(
            spike_times,
            event_onset_times,
            [baseline_start, evoked_end],
            spikeindex=False
        )

        # Convert to arrays
        spikeTimesFromEventOnset = np.array(spikeTimesFromEventOnset)
        trialIndexForEachSpike = np.array(trialIndexForEachSpike)
        indexLimitsEachTrial = np.array(indexLimitsEachTrial)

        # Converts spike times into spike counts in a matrix
        spikeCountMatrix = jaratoolbox.spikesanalysis.spiketimes_to_spikecounts(spikeTimesFromEventOnset,
                                                                                indexLimitsEachTrial,
                                                                                full_time)

        spikeCountMatrix = np.array(spikeCountMatrix)

        # Get the corresponding frequencies for each event onset time from the behavior data (the current frequency presented)
        freqEachTrial = behavior_data['currentFreq']
        possibleFreq = np.unique(freqEachTrial)
        if session_type == "pureTones":
            freqLabels = ['{0:.0f}'.format(freq / 1000) for freq in possibleFreq]
        else:
            freqLabels = ['{0:.0f}'.format(freq) for freq in possibleFreq]
        trialsEachType = jaratoolbox.behavioranalysis.find_trials_each_type(freqEachTrial, possibleFreq)
        cell_category = None

        # pRaster, hCond, zline = extraplots.raster_plot(spike_times, indexLimitsEachTrial,
        # timeRange, trialsEachType, labels=freqLabels)

        # Plot raster
        plt.subplot(2, 2, 2 - i)
        # Define Tango color numbers
        custom_colors = np.array(
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf',
             '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d',
             '#9edae5'])
        raster_colors = [custom_colors[cond] for cond in range(len(possibleFreq))]
        raster = extraplots.raster_plot(spikeTimesFromEventOnset, indexLimitsEachTrial, timeRange=timeRange,
                                        trialsEachCond=trialsEachType,
                                        colorEachCond=raster_colors, fillWidth=None, labels=freqLabels, rasterized=True)

        plt.title(f'{session_type} - Raster Plot')
        plt.xlabel('Time from event onset (s)')
        if session_type == "pureTones":
            plt.ylabel('Frequency (kHz)')
            y_ticks = np.round(plt.yticks()[0], 1)
            plt.yticks(y_ticks)
        else:
            plt.ylabel('AM Rate (Hz)')

        # Plot PSTH
        plt.subplot(2, 2, 4 - i)
        psth_colors = [custom_colors[cond] for cond in range(len(possibleFreq))]
        binsStartTime = np.arange(timeRange[0], timeRange[1], binWidth)

        psth = extraplots.plot_psth(spikeCountMatrix / binWidth, smoothWinSize=6, binsStartTime=binsStartTime,
                                    trialsEachCond=trialsEachType,
                                    colorEachCond=psth_colors, linestyle=None, linewidth=3, downsamplefactor=1)

        plt.title(f'{session_type} - PSTH')
        plt.xlabel('Time from event onset (s)')
        plt.ylabel('Firing Rate (spikes/s)')

    plt.tight_layout()
    plt.suptitle(f'Cell {cell_index}', fontsize=14)

    folder_path = '/Users/zoetomlinson/Desktop/NeuroAI/self_exploration/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    #plt.savefig(os.path.join(folder_path, f'Cell {cell_index}.png'))
    plt.show()

    # If you need to change the markersize on the raster use the following:
    # plt.setp(pRaster, ms=figparams.rasterMS)

    # The following code is for adding boxes for stim duration (bbox) or brackets (arrowprops) for showing regions of focus on the raster
    # ax = pRaster[0].axes

    '''ax.annotate('Evoked Response', xy=(0.51, 1.01), xytext=(0.51, 1.05),
                fontsize=12, fontweight='bold', ha='center',
                va='bottom', xycoords='axes fraction',
                bbox=dict(boxstyle='square', fc='0.8'),
                arrowprops=dict(arrowstyle='-[, widthB=1.3, lengthB=0.5', lw=2.0, color='k'))'''
