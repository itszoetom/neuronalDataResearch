import numpy as np
import matplotlib.pyplot as plt
import os
import jaratoolbox
from jaratoolbox import celldatabase, ephyscore, settings
from sklearn.decomposition import PCA

# Calculate Spike Rate and Visualizes PCA on two cells. One mouse, one day.

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
plt.scatter(cell_1_PTSpikeRate, cell_2_PTSpikeRate)
plt.title('Pure Tones')
plt.xlabel(f'Cell {cell_1} Spike Rate (spikes/s)')
plt.ylabel(f'Cell {cell_2} Spike Rate (spikes/s)')
# plt.colorbar(label='Pure Tones')

# Plotting for AM
plt.subplot(1, 2, 2)
plt.scatter(cell_1_AMSpikeRate, cell_2_AMSpikeRate)
plt.title('AM')
plt.xlabel(f'Cell {cell_1} Spike Rate (spikes/s)')
plt.ylabel(f'Cell {cell_2} Spike Rate (spikes/s)')
# plt.colorbar(label='AM Frequency (Hz)')

# PCA Analysis
x = np.vstack((cell_1_AMSpikeRate, cell_2_AMSpikeRate)).T

# Subtract the mean from the data:
x_mean = np.mean(x, axis=0)
x_meansub = x - x_mean

# Perform PCA on the data:
pca = PCA()
pca.fit(x_meansub)
PCs = pca.components_  # each row is one of the PC vectors

pc1 = PCs[0, :]
pc2 = PCs[1, :]

print('PC vectors have length=1: ', np.sqrt(pc1[0]**2 + pc1[1]**2), np.sqrt(pc2[0]**2 + pc2[1]**2))
print('PC vectors are orthogonal to one another: ', np.dot(pc1, pc2))

plt.figure(figsize=(5, 5))
plt.plot(x[:, 0], x[:, 1], 'o')
plt.plot([0, pc1[0]], [0, pc1[1]], lw=3)
plt.plot([0, pc2[0]], [0, pc2[1]], lw=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['data', 'PC1', 'PC2'])

# How much variance is explained by each PC?
print('Variance explained by (PC1, PC2): ', pca.explained_variance_)
print('Fraction of variance explained by (PC1, PC2): ', pca.explained_variance_ratio_)

std1 = pca.explained_variance_[0]**0.5
std2 = pca.explained_variance_[1]**0.5
pc1_rescaled = std1 * pc1
pc2_rescaled = std2 * pc2

plt.figure(figsize=(5, 5))
plt.plot(x_meansub[:, 0], x_meansub[:, 1], 'o', alpha=0.3)
plt.plot([-pc1_rescaled[0], pc1_rescaled[0]],
         [-pc1_rescaled[1], pc1_rescaled[1]], lw=3)
plt.plot([-pc2_rescaled[0], pc2_rescaled[0]],
         [-pc2_rescaled[1], pc2_rescaled[1]], lw=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['data', 'PC1', 'PC2'])

# Project the data onto the principal component axes:
proj = x_meansub @ PCs.T  # Each row gives the coordinates of a data point along the PC axes.

# Plot the data projected along the first two PCs:
plt.figure(figsize=(5, 5))
plt.plot(proj[:, 0], proj[:, 1], 'o')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.tight_layout()
plt.show()
