"""
Split results of multisession spike-sorting back to individual sessions.

Automated: processes all valid multisession folders for subjects feat004–feat010,
skipping any folders that end in "_prephy".
"""

import os
import shutil
import re
from jaratoolbox import loadneuropix
from jaratoolbox import settings
import importlib

# Reload to ensure latest version
importlib.reload(loadneuropix)

# ----- CONFIG -----
debug = False  # Set to True to simulate processing without saving files
base_path = '/Volumes/NardociData/ephys/neuropixels'
allowed_subjects = [f'feat00{i}' for i in range(4, 10)] + ['feat010']
pattern = re.compile(r'multisession_(\d{4}-\d{2}-\d{2})_(\d+)um_processed')
# -------------------

scriptName = os.path.basename(__file__)
print(f"Running {scriptName} in {'DEBUG' if debug else 'NORMAL'} mode\n")

# Loop only through feat004–feat010
for subject in sorted(os.listdir(base_path)):
    if subject not in allowed_subjects:
        continue

    subject_path = os.path.join(base_path, subject)
    if not os.path.isdir(subject_path):
        continue

    print(f'Processing subject: {subject}')
    sessionsRootPath = os.path.join(settings.EPHYS_NEUROPIX_PATH, subject)

    for folder in sorted(os.listdir(sessionsRootPath)):
        if folder.endswith('_prephy'):
            continue  # Skip prephy folders

        match = pattern.match(folder)
        if not match:
            continue

        dateStr, depthStr = match.groups()
        pdepth = int(depthStr)

        multisessionProcessedDir = os.path.join(sessionsRootPath, folder)
        print(f' → Found multisession dir: {folder} (Date: {dateStr}, Depth: {pdepth}μm)')

        if debug:
            print('\nRunning in DEBUG mode. Messages will appear, but nothing will be created/saved.\n')

        # -- Extract and save spike shapes --
        if not debug:
            loadneuropix.spikeshapes_from_templates(multisessionProcessedDir, save=True)
        else:
            print('DEBUG: This would save cluster waveforms (spike shapes).\n')

        # -- Split spike times into sessions --
        sessionsList, sessionsDirs = loadneuropix.split_sessions(multisessionProcessedDir, debug=debug)

        # -- Copy Events and Info to each session --
        for inds, oneSessionProcessedDir in enumerate(sessionsDirs):
            subDir = os.path.join(multisessionProcessedDir, sessionsList[inds])
            if not debug:
                shutil.copytree(os.path.join(subDir, 'events'),
                                os.path.join(oneSessionProcessedDir, 'events'), dirs_exist_ok=True)
            print(f'Copied events to {oneSessionProcessedDir}/')

            if not debug:
                shutil.copytree(os.path.join(subDir, 'info'),
                                os.path.join(oneSessionProcessedDir, 'info'), dirs_exist_ok=True)
            print(f'Copied info to {oneSessionProcessedDir}/')

            multiDirFile = os.path.join(oneSessionProcessedDir, 'multisession_dir.txt')
            if not debug:
                with open(multiDirFile, 'w') as dirFile:
                    dirFile.write(multisessionProcessedDir)
            print(f'Saved {multiDirFile}\n')