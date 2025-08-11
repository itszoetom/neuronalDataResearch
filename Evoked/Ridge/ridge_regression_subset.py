import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from jaratoolbox import celldatabase, settings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import funcs
import seaborn as sns
import params
# %% Constants
subject_list = ['feat004', 'feat005', 'feat006', 'feat007', 'feat008', 'feat009']  # 'feat010'
recordingDate_list = {
    'feat004': ['2022-01-11', '2022-01-19', '2022-01-21'],
    'feat005': ['2022-02-07', '2022-02-08', '2022-02-11', '2022-02-14', '2022-02-15', '2022-02-16'],
    'feat006': ['2022-02-21', '2022-02-22', '2022-02-24', '2022-02-25', '2022-02-26', '2022-02-28', '2022-03-01', '2022-03-02'],
    'feat007': ['2022-03-10', '2022-03-11', '2022-03-15', '2022-03-16', '2022-03-18', '2022-03-21'],
    'feat008': ['2022-03-23', '2022-03-24', '2022-03-25'],
    'feat009': ['2022-06-04', '2022-06-05', '2022-06-06', '2022-06-07', '2022-06-09', '2022-06-10'],
    'feat010': ['2022-06-21', '2022-06-22', '2022-06-27', '2022-06-28', '2022-06-30']
}
targetSiteNames = ["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]
leastCellsArea = 10000
evoked_start = 0.015
evoked_end = 0.3
pt_evoked_end = 0.1
binWidth = 0.01
binEdges = np.arange(evoked_start, evoked_end, binWidth)
binEdgesPT = np.arange(evoked_start, pt_evoked_end, binWidth)
iterations = 30  # Number of random sampling iterations
random_seed = 42
alphas = np.logspace(-5, 10, 200)
np.random.seed(random_seed)  # Ensure reproducibility
periodsNameSpeech = ['base200', 'respOnset', 'respSustained']
max_trials = {'PT': 640, 'AM': 220, 'speech': 381}
unique_labels = [(0, 0), (0, 33), (0, 67), (0, 100), (33, 100), (67, 100), (100, 100), (100, 67), (100, 33),
                 (100, 0), (67, 0), (33, 0)]
min_speech_freq_dict = {(0, 0): 31, (0, 33): 29, (0, 67): 32, (0, 100): 24, (33, 100): 34, (67, 100): 35,
                        (100, 100): 33, (100, 67): 29, (100, 33): 35, (100, 0): 35, (67, 0): 31, (33, 0): 33}
# Initialize a dictionary to store counts for each frequency across mouse-date combos
frequency_counts_dict = {tuple(freq): [] for freq in unique_labels}
data_dict = {}
previous_frequency_speech = None
previous_frequency_AM = None
previous_frequency_PT = None
n_neurons_list = []
r2_test_list = []
labels_list = []
primary_n_neurons = 0
ventral_n_neurons = 0
dorsal_n_neurons = 0
indices_AM = None
results = []
smallest_neuron_count = 10
X_speech_all = []
Y_brain_area_speech_all = []
X_AM_all = []
Y_brain_area_AM_all = []
X_pureTones_all = []
Y_brain_area_PT_all = []


# %% Load Data
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
simpleSiteNames = simpleSiteNames.replace("Posterior auditory area", "Dorsal auditory area")
fullDb["recordingSiteName"] = simpleSiteNames

# Add data to the dictionary for each brain area and sound type
for subject in params.subject_list:
    for date in params.recordingDate_list[subject]:
        for brain_area in params.targetSiteNames:
            speechEnsemble, speechEphys, speechBdata = funcs.load_data(subject, date, brain_area, "FTVOTBorders")

            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = funcs.spike_rate("speech", speechEnsemble,
                                                                    speechEphys, speechBdata, brain_area)

                X_speech_array, Y_frequency_speech_sorted, previous_frequency_speech, indices_speech = (
                    funcs.adjust_speech_length(subject, date, brain_area, X_speech, Y_frequency_speech,
                                               Y_brain_area_speech, previous_frequency_speech))

                if X_speech_array is not None:
                    # Y_frequency_FT = Y_frequency_speech_sorted[:, 0]
                    # Y_frequency_VOT = Y_frequency_speech_sorted[:, 1]
                    X_speech_all.extend(X_speech_array)
                    Y_brain_area_speech_all.extend(Y_brain_area_speech)

            # Load and process data for AM
            amEnsemble, amEphys, amBdata = funcs.load_data(subject, date, brain_area, "AM")
            if amEnsemble:
                X_AM, Y_brain_area_AM, Y_frequency_AM = funcs.spike_rate(
                    "AM", amEnsemble, amEphys, amBdata, brain_area)

                # Apply adjustments
                X_AM_adjusted, Y_frequency_AM_adjusted, Yba_AM_adj = (
                    funcs.adjust_array_and_labels(X_AM, Y_frequency_AM, Y_brain_area_AM,
                                                  params.max_trials['AM'], subject, date, brain_area))

                # Sort arrays
                X_AM_array, Y_frequency_AM_sorted, Y_brain_area_AM_sorted, previous_frequency_AM, indices_AM = (
                    funcs.sort_sound_array(subject, date, brain_area, X_AM_adjusted, Yba_AM_adj, Y_frequency_AM_adjusted, previous_frequency_AM))

                if X_AM_array is not None:
                    # Append the data to the lists
                    X_AM_all.extend(X_AM_array)
                    Y_brain_area_AM_all.extend(Y_brain_area_AM_sorted)

            # Load and process data for Pure Tones
            ptEnsemble, ptEphys, ptBdata = funcs.load_data(subject, date, brain_area, "pureTones")
            if ptEnsemble:
                X_pureTones, Y_brain_area_PT, Y_frequency_pureTones = funcs.spike_rate(
                    "PT", ptEnsemble, ptEphys, ptBdata, brain_area)

                # Apply adjustments
                X_PT_adjusted, Y_frequency_PT_adjusted, Yba_PT_adj = (
                    funcs.adjust_array_and_labels(X_pureTones, Y_frequency_pureTones, Y_brain_area_PT,
                                                  params.max_trials['PT'], subject, date, brain_area))

                # Sort arrays
                X_PT_array, Y_frequency_PT_sorted, Y_brain_area_PT_sorted, previous_frequency_PT, indices_PT = (
                    funcs.sort_sound_array(subject, date, brain_area, X_PT_adjusted, Yba_PT_adj, Y_frequency_PT_adjusted, previous_frequency_PT))

                if X_PT_array is not None:
                    X_pureTones_all.extend(X_PT_array)
                    Y_brain_area_PT_all.extend(Y_brain_area_PT_sorted)

# %% Sort arrays
X_PT_sorted = funcs.sort_x_arrays(X_pureTones_all, indices_PT, "pt")
X_AM_sorted = funcs.sort_x_arrays(X_AM_all, indices_AM, "am")
X_speech_sorted = funcs.sort_x_arrays(X_speech_all, indices_speech, "speech")

# Convert the lists to numpy arrays for easy manipulation
X_speech_array = np.stack(X_speech_sorted, axis=0)
# X_speech_array = X_speech_array.squeeze(axis=1)
X_AM_array = np.stack(X_AM_sorted, axis=0)
X_PT_array = np.stack(X_PT_sorted, axis=0)

# Initialize a dictionary to store 2D arrays for each brain_area - sound_type combo
data_dict = {}

for subject in subject_list:
    for brain_area in params.targetSiteNames:
        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['AM', 'PT', 'speech'],
                [X_AM_array, X_PT_array, X_speech_array],
                [Y_brain_area_AM_all, Y_brain_area_PT_all, Y_brain_area_speech_all],
                [Y_frequency_AM_sorted, Y_frequency_PT_sorted, Y_frequency_speech_sorted]):

            # Filter data based on the brain_area
            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]  # X is neurons x trials

            # Store the 2D array (neurons x trials) in the dictionary
            data_dict[(subject, brain_area, sound_type)] = {
                'X': X_array_adjusted.T,  # Transpose to get the correct shape (neurons x trials)
                'Y': np.array(Y_frequency_sorted)}  # Y is a 1D array of frequencies

# %% Ridge Regression
for key, value in data_dict.items():
    X = value['X']
    if (key[0] == 'feat004' or key[0] == 'feat006') and (key[1] == 'Primary auditory area' or key[1] == 'Ventral auditory area'):
        continue  # dont include this data because there arent enough neurons
    if (key[0] == 'feat008' or key[0] == 'feat009') and key[1] == 'Dorsal auditory area':
        continue
    smallest_neuron_count = 60

    if key[1] == 'AM' or key[1] == 'PT':
        Y = np.log(value['Y'])
    else:
        Y = value['Y']

    # Sampling iterations
    sampled_indices = np.random.choice(X.shape[1], smallest_neuron_count, replace=False)
    X_sampled = X[:, sampled_indices]
    n_neurons = X_sampled.shape[1]

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_sampled, Y, test_size=0.2,
                                                        random_state=random_seed)
    # Check if the array is empty before fitting the model
    if X_train.shape[1] == 0:
        print(f"Skipping {key[0]} - {key[1]} - {key[2]} due to empty feature array.")
        continue  # Skip this iteration

    # Find best alpha using Ridge Regression
    best_r2, best_alpha = -np.inf, None
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, Y_train)
        r2 = ridge.score(X_test, Y_test)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    # Append results
    results.append({
        'Subject': key[0],
        'Brain Area': key[1],
        'Sound Type': key[2],
        'R2 Test': best_r2
        })

    if key[1] == "Primary auditory area":
        primary_n_neurons += n_neurons
    elif key[1] == "Dorsal auditory area":
        dorsal_n_neurons += n_neurons
    else:
        ventral_n_neurons += n_neurons

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Compute the average R² for each Subject, Brain Area, and Sound Type
average_r2_df = results_df.groupby(['Subject', 'Brain Area', 'Sound Type'], as_index=False)['R2 Test'].mean()
from matplotlib import cm

# Define consistent order for sound types
sound_order = ['PT', 'AM', 'speech']

# Map sound types to specific colors from different colormaps
sound_cmaps = {
    "PT": cm.winter(0.5),
    "AM": cm.magma(0.5),
    "speech": cm.summer(0.5),
}

# Plot with white background and color mapping
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=average_r2_df,
    x='Brain Area',
    y='R2 Test',
    hue='Sound Type',
    order=["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"],
    hue_order=sound_order,
    palette=[sound_cmaps[sound] for sound in sound_order],
    ax=ax
)

# Style the plot
plt.title("R² Scores by Brain Area and Sound Type", fontsize=18)
plt.xlabel("Brain Area", fontsize=14)
plt.ylabel("Correlation (R² Score)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Sound Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig("/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Ridge Regression/R2_BoxPlot_Poster.png")
plt.show()