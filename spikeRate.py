import numpy as np
import matplotlib.pyplot as plt
import os
import jaratoolbox
from jaratoolbox import celldatabase, ephyscore, settings

# Loading Data Calculating Spike Rate for Pure Tones and Amplitude Modulation Sounnds. Neurons from one mouuse, one day

# Loads in dataframe for one mouse
subject = 'feat004'
studyName = '2022paspeech'
dbPath = os.path.join(settings.DATABASE_PATH, studyName, f'celldb_{subject}.h5')
oneMouseDf = celldatabase.load_hdf(dbPath)

sessionDate = '2022-01-11'
probeDepth = 2318
celldbSubset = oneMouseDf[(oneMouseDf.date == sessionDate) & (oneMouseDf.pdepth == probeDepth)]

ensemble = jaratoolbox.ephyscore.CellEnsemble(celldbSubset)

# Sets time ranges for plotting and calculating firing rates
behavior_class = None
baseline_start = -0.1
baseline_end = 0.3
evoked_start = 0.015
evoked_end = 0.3
binWidth = 0.01
binEdges = np.arange(evoked_start, evoked_end, binWidth)

# Sets what cells you are graphing
cell_1 = 21
cell_2 = 23


# PT Code
def spikeRate(sessionType):
    ephysData, bdata = ensemble.load(sessiontype=sessionType, behavClass=behavior_class)

    # Gets spike times and event onset times (i.e. when the sounds were presented or the stim turns on)
    spike_times = ephysData['spikeTimes']
    nTrials = len(bdata['currentFreq'])
    eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]

    # Lines up event times and spike times
    spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial = ensemble.eventlocked_spiketimes(
        eventOnsetTimes,
        [evoked_start, evoked_end])

    spikeCountMat = ensemble.spiketimes_to_spikecounts(binEdges)
    spikeCountMat = spikeCountMat.sum(axis=2)
    #spikeCountMat = spikeCountMat[:-1]
    spike_rate = spikeCountMat / (evoked_end - evoked_start)
    colormap = np.where(bdata['currentFreq'] > bdata['currentFreq'].mean(), "red", "blue")

    return spike_rate, colormap

# Plotting for PT
PTSpikeRate, PTcolormap = spikeRate("pureTones")

cell_1_PTSpikeRate = PTSpikeRate[cell_1, :]
cell_2_PTSpikeRate = PTSpikeRate[cell_2, :]

# Plotting for AM
AMSpikeRate, AMcolormap = spikeRate("AM")

cell_1_AMSpikeRate = AMSpikeRate[cell_1, :]
cell_2_AMSpikeRate = AMSpikeRate[cell_2, :]

# Plotting
plt.figure(figsize=(10, 5))

# Plotting for Pure Tones
plt.subplot(1, 2, 1)
plt.scatter(cell_1_PTSpikeRate, cell_2_PTSpikeRate, c=PTcolormap)
plt.title('Pure Tones')
plt.xlabel(f'Cell {cell_1} Spike Rate (spikes/s)')
plt.ylabel(f'Cell {cell_2} Spike Rate (spikes/s)')
# plt.colorbar(label='Pure Tones')

# Plotting for AM
plt.subplot(1, 2, 2)
plt.scatter(cell_1_AMSpikeRate, cell_2_AMSpikeRate, c=AMcolormap)
plt.title('AM')
plt.xlabel(f'Cell {cell_1} Spike Rate (spikes/s)')
plt.ylabel(f'Cell {cell_2} Spike Rate (spikes/s)')
# plt.colorbar(label='AM Frequency (Hz)')

plt.tight_layout()
plt.show()
