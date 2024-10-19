#%%
import os
import studyparams
import numpy as np
import pandas as pd
from jaratoolbox import celldatabase
from jaratoolbox import settings
from jaratoolbox import spikesanalysis
from jaratoolbox import ephyscore
from jaratoolbox import behavioranalysis
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
from jaratoolbox import extraplots

#%% Loading test database

subject = 'feat007'
databaseDir = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)  # Note: Change these to use PathLib instead
dbPath = os.path.join(databaseDir, f'{subject}_paspeech_am_tuning.h5')

mouseDB = celldatabase.load_hdf(dbPath)

#%% Number of sessions

total_recordings = mouseDB.date.unique().size  # Using the general assumption that one date = one recording
print(f"Mouse {subject} has {total_recordings} recordings conducted\n")

# Counting how many instances of session type exist for each unique value, if there are any
sessions = mouseDB.sessionType.value_counts()
session_keys = sessions.keys()[0]  # yoinking keys for label purposes later
paradigms = mouseDB.paradigm.value_counts()
paradigm_keys = paradigms.keys()[0]  # yoinking paradigm keys for matching
print(f"These recordings include {sessions[0]} sessions of {session_keys[0]}, {session_keys[1]}, and {session_keys[2]}"
      f" presentations from the {paradigm_keys[0]}, {paradigm_keys[1]}, and {paradigm_keys[2]} paradigms respectively\n")

# Seems a bit odd that there is the same paradigm twice for different sessions...


# ---------------------------- Plotting code STARTO ----------------------------------
#%% Loading population data for a session

recordingDate = mouseDB.date.unique()[0]
celldbSubset = mouseDB[mouseDB.date == recordingDate]

ensemble = ephyscore.CellEnsemble(celldbSubset)
ephysDataSpeech, bdataSpeech = ensemble.load("FTVOTBorders")
# ephysDataAM, bdataAM = ensemble.load("AM")
# ephysDataPT, bdataPT = ensemble.load("pureTones")

# Loading ephys data
spikeTimes = ephysDataSpeech['spikeTimes']
eventOnsetTimes = ephysDataSpeech['events']['stimOn']
timeRange = [-0.4, 0.55]  # In seconds
timeRangeSpontan = [-0.3, -0.1]  # 200 ms of spontaneous firing
timeRangeEvoked = [0, 0.2]  # 200 ms of evoked firing

# Aligning spike data to event times
spikeTimesFromEventOnsetAll, trialIndexForEachSpikeAll, indexLimitsEachTrialAll = \
    ensemble.eventlocked_spiketimes(eventOnsetTimes, timeRange)

#%% bdata stuff
# Grabbing the formant transition values presented (given as a percent, 0 = ba 1 = da)
FTParamsEachTrial = bdataSpeech['targetFTpercent']
possibleFTParams = np.unique(FTParamsEachTrial)
nTrials = len(FTParamsEachTrial)

# grabbing the Voice Onset Time values presented (given as a percent, 0 = ba 1 = pa)
VOTParamsEachTrial = bdataSpeech['targetVOTpercent']
possibleVOTParams = np.unique(VOTParamsEachTrial)

# One hot encoded matrix for which stim was presented for the trial
# trialsEachCond = behavioranalysis.find_trials_each_type(FTParamsEachTrial, possibleFTParams)

trialsEachCond = behavioranalysis.find_trials_each_combination(VOTParamsEachTrial, possibleVOTParams, FTParamsEachTrial,
                                                               possibleFTParams)

condEachSortedTrialVOT, condEachSortedTrialFT, sortedTrialsIndex = np.nonzero(trialsEachCond.T)

# Getting spike times
binSize = 0.005  # 5 ms spike time bins
binEdges = np.arange(timeRange[0], timeRange[1], binSize)
spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)  # (nCells, nTrials, nBins)
nCells = spikeCounts.shape[0]

sortedSpikeCount = spikeCounts[:, sortedTrialsIndex, :]

#%% Initial raster to see if data looks as expected
figRas = plt.figure(figsize=[10, 4])
cellToPlot = 13
plt.imshow(sortedSpikeCount[cellToPlot], aspect='auto', origin='lower')
plt.xlabel('Time bin')
plt.ylabel('Trial (sorted by stim type)')
plt.axvline(x=80)
plt.colorbar(label='N spikes')
plt.show()

#%% Continued sorting of data
# Initially taking 4 corners of data for box-and-whisker plot
trialsEachVOTmax_FTmin = trialsEachCond[:, -1, 0]
trialsEachVOTmin_FTmin = trialsEachCond[:, 0, 0]
trialsEachVOTmax_FTmax = trialsEachCond[:, -1, -1]
trialsEachVOTmin_FTmax = trialsEachCond[:, 0, -1]

# trials when one stim is maxed/mined and the other varies (rows or column of border)
# trialsEachVOT_FTmin = trialsEachCond[:, :, 0]
# trialsEachVOT_FTmax = trialsEachCond[:, :, -1]
# trialsEachFT_VOTmin = trialsEachCond[:, 0, :]
# trialsEachFT_VOTmax = trialsEachCond[:, -1, :]

# Box-and-whisker
popFR = spikeCounts.mean(axis=2)  # shape is (nCells, nTrials), averaging FR of bins

catTrials = []
nCornerTrials = 0

cornerLabels = ["", "pa", "ba", "ta", "da"]
for idx in [trialsEachVOTmax_FTmin, trialsEachVOTmin_FTmin, trialsEachVOTmax_FTmax, trialsEachVOTmin_FTmax]:
    targetTrials = popFR[:, idx].mean(axis=0)
    # nCornerTrials += targetTrials.shape[1]
    catTrials.append(targetTrials)

# catTrials = catTrials.reshape([nCells, nCornerTrials])

#%%
# it's plotting time
figBP, axBP = plt.subplots(figsize=(8, 6))
bp = plt.boxplot(catTrials, notch=True, bootstrap=2000)
axBP.set_xlim(0,5)
plt.ylabel("Firing rate (spk/s)")
plt.xlabel("Phoneme")
plt.xticks(np.arange(0, 5), cornerLabels)
plt.show()

#%% Plotting cell locations
locations = mouseDB.recordingSiteName
locCount = len(locations.unique())

figLoc, axLoc = plt.subplots(figsize=(12,10))
axLoc.set_xticklabels(locations.unique(), rotation=45, ha="right")
plt.subplots_adjust(bottom=0.2)
hist = plt.hist(locations, bins=locCount, rwidth=0.9)
axLoc.set_ylabel("Number of cells")
plt.show()

#%% Actual population calculations - Evoked firing speech
FTParamsEachTrial = bdataSpeech['targetFTpercent']
possibleFTParams = np.unique(FTParamsEachTrial)
nTrials = len(FTParamsEachTrial)

# grabbing the Voice Onset Time values presented (given as a percent, 0 = ba 1 = pa)
VOTParamsEachTrial = bdataSpeech['targetVOTpercent']
possibleVOTParams = np.unique(VOTParamsEachTrial)

# One hot encoded matrix for which stim was presented for the trial
# trialsEachCond = behavioranalysis.find_trials_each_type(FTParamsEachTrial, possibleFTParams)

trialsEachCond = behavioranalysis.find_trials_each_combination(VOTParamsEachTrial, possibleVOTParams, FTParamsEachTrial,
                                                               possibleFTParams)

condEachSortedTrialVOT, condEachSortedTrialFT, sortedTrialsIndex = np.nonzero(trialsEachCond.T)
sortingInds = np.argsort(sortedTrialsIndex)

# Getting spike times
binSize = 0.005  # 5 ms spike time bins
binEdges = np.arange(timeRangeEvoked[0], timeRangeEvoked[1], binSize)
spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)  # (nCells, nTrials, nBins)
nCells = spikeCounts.shape[0]

sortedSpikeCountEvoked = spikeCounts[:, sortedTrialsIndex, :]

avgFRTrial = sortedSpikeCountEvoked.mean(axis=1)
sumEvokedFR = avgFRTrial.sum(axis=1)
spikesPerSecEvoked = sumEvokedFR/(timeRangeEvoked[1] - timeRangeEvoked[0])

#%% Plotting evoked FR

figEvok, axEvok = plt.subplots(figsize=(7, 5))
histEvok = plt.hist(spikesPerSecEvoked, bins=100)
plt.ylabel("Number of cells")
plt.xlabel("Firing rate (spk/s)")
plt.title("Evoked firing rate averaged across ALL speech trials")
plt.axvline(x=1, color="orange", alpha=0.8)

plt.show()

#%% Actual population calculations - Spontaneous firing speech
# Getting spike times
binEdges = np.arange(timeRangeSpontan[0], timeRangeSpontan[1], binSize)
spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)  # (nCells, nTrials, nBins)
nCells = spikeCounts.shape[0]

sortedSpikeCountSpontan = spikeCounts[:, sortedTrialsIndex, :]

avgFRTrial = sortedSpikeCountSpontan.mean(axis=1)
sumSpontanFR = avgFRTrial.sum(axis=1)
spikesPerSecSpontan = sumSpontanFR/(timeRangeSpontan[1] - timeRangeSpontan[0])

#%% Plotting spontaneous firing rates
figSpon, axSpon = plt.subplots(figsize=(7, 5))
histSpon = plt.hist(spikesPerSecSpontan, bins=100, alpha=1)
plt.ylabel("Number of cells")
plt.xlabel("Firing rate (spk/s)")
plt.xlim([-2, 42])
plt.title("Spontaneous firing rate averaged across ALL speech trials")
plt.axvline(x=1, color="black", alpha=0.8)

plt.show()

#%% Overlapped
figCombined, axCombined = plt.subplots(figsize=(7, 5))
histSponComb = plt.hist(spikesPerSecSpontan, bins=50, alpha=0.5)
histEvokComb = plt.hist(spikesPerSecEvoked, bins=50, alpha=0.5)
plt.ylabel("Number of cells")
plt.xlabel("Firing rate (spk/s)")
plt.xlim([-2, 42])
plt.title("Firing rate averaged across ALL speech trials")
plt.legend(["Spontaneous", "Evoked"], loc="best")
plt.axvline(x=spikesPerSecSpontan.mean(), color="blue", alpha=1)
plt.axvline(x=spikesPerSecEvoked.mean(), color="orange", alpha=1)
plt.show()

#%% Separating out by the 4 corners of the speech stim

trialsEachCond = behavioranalysis.find_trials_each_combination(VOTParamsEachTrial, possibleVOTParams, FTParamsEachTrial,
                                                               possibleFTParams)

trialsEachVOTmax_FTmin = trialsEachCond[:, -1, 0]  # Pa
trialsEachVOTmin_FTmin = trialsEachCond[:, 0, 0]  # Ba
trialsEachVOTmax_FTmax = trialsEachCond[:, -1, -1]  # Ta
trialsEachVOTmin_FTmax = trialsEachCond[:, 0, -1]  # Da
trialLabels = ["pa", "ba", "ta", "da"]
labelIndex = 0

for idx in [trialsEachVOTmax_FTmin, trialsEachVOTmin_FTmin, trialsEachVOTmax_FTmax, trialsEachVOTmin_FTmax]:

    # Getting spike times
    binSize = 0.005  # 5 ms spike time bins
    binEdgesSpon = np.arange(timeRangeSpontan[0], timeRangeSpontan[1], binSize)
    binEdgesEvoked = np.arange(timeRangeEvoked[0], timeRangeEvoked[1], binSize)
    spikeCountsSpon = ensemble.spiketimes_to_spikecounts(binEdgesSpon)  # (nCells, nTrials, nBins)
    spikeCountsEvoked = ensemble.spiketimes_to_spikecounts(binEdgesEvoked)  # (nCells, nTrials, nBins)

    sortedSpikeCountSpon = spikeCountsSpon[:, np.where(idx == 1), :]  #TODO: Identify why an extra dimension gets added and the best way to collapse it. Probably just result.squeeze()
    sortedSpikeCountSpon = sortedSpikeCountSpon.squeeze()
    sortedSpikeCountEvoked = spikeCountsEvoked[:, np.where(idx == 1), :]
    sortedSpikeCountEvoked = sortedSpikeCountEvoked.squeeze()

    popFRinitSpon = sortedSpikeCountSpon.sum(axis=2)  # shape is (nCells, nTrials), averaging FR of bins
    popFRinitEvoked = sortedSpikeCountEvoked.sum(axis=2)  # shape is (nCells, nTrials), averaging FR of bins

    # Now averaging across trials and then dividing by the time bins to get spks/sec from the total counts
    avgSpikeCountSpon = popFRinitSpon.mean(axis=1)
    avgSpikeCountEvoked = popFRinitEvoked.mean(axis=1)

    popFRSpon = avgSpikeCountSpon/(timeRangeSpontan[1] - timeRangeSpontan[0])
    popFREvoked = avgSpikeCountEvoked/(timeRangeEvoked[1] - timeRangeEvoked[0])

    figContext, axContext = plt.subplots(figsize=(7, 5))
    histSponContext = plt.hist(popFRSpon, bins=50, alpha=0.5)
    histEvokContext = plt.hist(popFREvoked, bins=50, alpha=0.5)
    plt.ylabel("Number of cells")
    plt.xlabel("Firing rate (spk/s)")
    plt.xlim([-2, 42])
    plt.title(f"Firing rate averaged across {trialLabels[labelIndex]} trials")
    plt.legend(["Spontaneous", "Evoked"], loc="best")
    plt.axvline(x=popFRSpon.mean(), color="blue", alpha=1)  # TODO: Add in a text box on the right of the graph with the mean values
    plt.axvline(x=popFREvoked.mean(), color="orange", alpha=1)
    plt.savefig(f"/Users/Matt/Desktop/Research/Murray/data/images/20230706_population_histograms/histogram_population_{trialLabels[labelIndex]}_trials")
    plt.show()

    labelIndex += 1
    #%%
    cornerLabels = ["", "pa", "ba", "ta", "da"]
    # Produces a list with 4 arrays. Each array represents the cells summed spike counts
    # The order of sounds is pa, ba, ta, da within the list


#%% Attempt at iterative plotting

