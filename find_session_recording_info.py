"""
Utilize the bdata files to identify how many times each stimulus was presented. Possibly store this as a dict
 Each row in this frame should be one session from one recording for one mouse. This way unique dates for a mouse
 will inform how many recordings there were, unique ephysTime will say how many sessions for each recording,
 nCells is how many cells for each session, recordingSiteName covers how many cells per area, and nStim will
 be the dictionary describing number of each stimulus combo given. Idk the best way to quantify this tbh, but it's a
 start
"""

#%% Imports
import os
import studyparams
import numpy as np
import pandas as pd
from jaratoolbox import celldatabase
from jaratoolbox import settings
from jaratoolbox import ephyscore

#%% Creating empty frame
infoFrame = pd.DataFrame(columns=["subject", "date", "ephysTime", "sessionType", "nCells", "nStims"])

#%% Looping STARTO
databaseDir = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)  # Note: Change these to use PathLib instead

for subject in studyparams.EPHYS_MICE:
    dbPath = os.path.join(databaseDir, f'{subject}_paspeech_am_tuning.h5')
    mouseDB = celldatabase.load_hdf(dbPath)

    for date in mouseDB.date.unique():
        celldbSubset = mouseDB[mouseDB.date == date]

        for ephysIndex, session in enumerate(celldbSubset.sessionType.iat[0]):
            # Take the recording sites and split by a comma, then use a lambda function to take the output from split and store
            #  only the first part
            simpleSiteNames = celldbSubset['recordingSiteName'].str.split(',').apply(lambda x: x[0]).value_counts().to_dict()
            infoFrame = infoFrame.append({"simpleSiteName": simpleSiteNames}, ignore_index=True)

            infoFrame["date"].iat[-1] = date
            infoFrame["subject"].iat[-1] = subject
            infoFrame["sessionType"].iat[-1] = session
            infoFrame["nCells"].iat[-1] = celldbSubset.shape[0]
            infoFrame["ephysTime"].iat[-1] = celldbSubset.ephysTime.iat[0][ephysIndex]

            ensemble = ephyscore.CellEnsemble(celldbSubset)
            if session == "FTVOTBorders":
                ephysDataSpeech, bdataSpeech = ensemble.load(session)

                # Speech stuff
                FTParamsEachTrial = bdataSpeech['targetFTpercent']  # Can use np.unique(arr, return_counts=True) to get occurance of stim
                possibleFTParams = np.unique(FTParamsEachTrial)
                VOTParamsEachTrial = bdataSpeech['targetVOTpercent']
                possibleVOTparams = np.unique(VOTParamsEachTrial)

                tFrame = pd.DataFrame({"VOT": VOTParamsEachTrial, "FT": FTParamsEachTrial})
                speechCounts = tFrame[["VOT", "FT"]].value_counts(sort=False)
                infoFrame["nStims"].iat[-1] = speechCounts.to_dict()

            elif session == "AM":
                ephysDataAM, bdataAM = ensemble.load(session)
                # Currently only looking at frequency, ignoring intensity
                AMParamsEachTrial = bdataAM["currentFreq"]
                possibleAMParams, AMParamCounts = np.unique(AMParamsEachTrial, return_counts=True)
                AMDict = dict(zip(possibleAMParams, AMParamCounts))
                infoFrame["nStims"].iat[-1] = AMDict

            elif session == "pureTones":
                ephysDataPT, bdataPT = ensemble.load(session)
                # Currently only looking at frequency, ignoring intensity
                PTParamsEachTrial = bdataPT["currentFreq"]
                possiblePTParams, PTParamCounts = np.unique(PTParamsEachTrial, return_counts=True)
                PTDict = dict(zip(possiblePTParams, PTParamCounts))
                infoFrame["nStims"].iat[-1] = PTDict

            else:
                print(f"Session {session} does not fall in the recognized categories. Oops")
