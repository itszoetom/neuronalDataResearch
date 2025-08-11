import numpy as np
import matplotlib.pyplot as plt
import os
import jaratoolbox
from jaratoolbox import celldatabase, ephyscore, settings
from sklearn.decomposition import PCA

# Calculate and analyze spike rates for all Pure Tone and Amplitude Modulated sounds.
# Perform PCA on the ensemble data, visualize the principal components, and show the variance
# explained across all cells.

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
    spike_rate = spikeCountMat / (evoked_end - evoked_start)
    colormap = np.where(bdata['currentFreq'] > bdata['currentFreq'].mean(), "red", "blue")

    return spike_rate


# Visualization of PCA Results for all cells for both "pureTones" and "AM"
plt.figure(figsize=(12, 5))

for i, session_type in enumerate(["pureTones", "AM"]):
    sr = spikeRate(session_type)
    sr = sr.T

    # Perform PCA on the data:
    pca = PCA()
    pca.fit((sr - np.mean(sr, axis=0)))
    PCs = pca.components_  # each row is one of the PC vectors

    # How much variance is explained by each PC?
    print(f'Variance explained by {session_type}: ', pca.explained_variance_)
    print(f'Fraction of variance explained by {session_type}: ', pca.explained_variance_ratio_)
    print(sr.shape)

    # Plotting
    ax = plt.subplot(1, 2, i + 1)
    proj_cell = PCs @ ((sr - np.mean(sr, axis=0)).T)
    ax.plot(proj_cell[:, 0], proj_cell[:, 1], 'o', alpha = 0.5)
    plt.title(f'PCA Results for {session_type}')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.plot(pca.explained_variance_ratio_[:25])


plt.tight_layout()
plt.show()
