import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from jaratoolbox import celldatabase, settings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import funcs
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# %% Constants
subject_list = ['feat004', 'feat005', 'feat006', 'feat007', 'feat008', 'feat009', 'feat010']  # ,
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
periodsNameSpeech = ['base200', 'respOnset', 'respSustained']
max_trials = {'PT': 640, 'AM': 220, 'speech': 381}
unique_labels = [(0, 0), (0, 33), (0, 67), (0, 100), (33, 100), (67, 100), (100, 100), (100, 67), (100, 33),
                 (100, 0), (67, 0), (33, 0)]
min_speech_freq_dict = {(0, 0): 31, (0, 33): 29, (0, 67): 32, (0, 100): 24, (33, 100): 34, (67, 100): 35,
                        (100, 100): 33, (100, 67): 29, (100, 33): 35, (100, 0): 35, (67, 0): 31, (33, 0): 33}
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
primary_am_neurons = 0
primary_pt_neurons = 0
ventral_am_neurons = 0
ventral_pt_neurons = 0

# %% Load dataframe
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
simpleSiteNames = simpleSiteNames.replace("Posterior auditory area", "Dorsal auditory area")
fullDb["recordingSiteName"] = simpleSiteNames

# Add data to the dictionary for each brain area and sound type
for subject in subject_list:
    for brain_area in targetSiteNames:

        X_speech_array, X_AM_array, X_PT_array, \
            Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all, \
            Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted \
            = funcs.clean_and_concatenate(subject, recordingDate_list, brain_area,
                                          previous_frequency_AM, previous_frequency_PT,
                                          previous_frequency_speech)

        Y_frequency_FT = Y_frequency_speech_sorted[:,0]
        Y_frequency_VOT = Y_frequency_speech_sorted[:, 1]

        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['AM', 'PT', 'FT', 'VOT'],
                [X_AM_array, X_PT_array, X_speech_array, X_speech_array],
                [Y_brain_area_AM_all, Y_brain_area_PT_all, Y_brain_area_speech_all, Y_brain_area_speech_all],
                [Y_frequency_AM_sorted, Y_frequency_pureTones_sorted, Y_frequency_FT, Y_frequency_VOT]):

            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]
            X_array_adjusted = X_array_adjusted.T
            data_dict[(subject, brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

# 5% tolerance of the target range
tolerance = 0.05
# Define the alpha range
alphas = np.logspace(-10, 5, 600)

for key, value in data_dict.items():
    X = value['X']
    if key[2] == 'AM' or key[2] == 'PT':
        Y = np.log(value['Y'])
    else:
        Y = value['Y']

    # Skip empty feature arrays early
    if X.shape[1] == 0:
        print(f"Skipping {key[0]} - {key[1]} - {key[2]} due to empty feature array.")
        continue

    n_neurons = X.shape[1]
    n_neurons_list.append(n_neurons)  # Only append when not skipped

    if key[1] == 'Primary auditory area':
        primary_n_neurons += n_neurons
    elif key[1] == 'Dorsal auditory area':
        dorsal_n_neurons += n_neurons
    else:
        ventral_n_neurons += n_neurons

    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Cross-validation to select the best alpha
    best_r2 = -np.inf
    best_alpha_idx = None
    for i, alpha in enumerate(alphas):
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train, Y_train)
        r2_score = ridge_reg.score(X_test, Y_test)
        if r2_score > best_r2:
            best_r2 = r2_score
            best_alpha_idx = i

    # Append R^2 values only when valid
    r2_test_list.append(best_r2)
    labels_list.append(f"{key[0]} - {key[1]} - {key[2]}")

    # Plot for best model visualization
    best_alpha = alphas[best_alpha_idx]
    ridge_reg = Ridge(alpha=best_alpha)
    ridge_reg.fit(X_train, Y_train)

    # Predict and evaluate
    y_train_pred = ridge_reg.predict(X_train)
    y_test_pred = ridge_reg.predict(X_test)
    tolerance_range = tolerance * Y_test
    within_tolerance = np.abs(y_test_pred - Y_test) <= tolerance_range
    percent_within_tolerance = np.mean(within_tolerance) * 100

    # Sort test data and predicted values based on the test data
    sorted_indices = np.argsort(Y_test)
    Y_test_sorted = Y_test[sorted_indices]
    y_test_pred_sorted = y_test_pred[sorted_indices]

    # Define Metrics
    rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
    corr, _ = pearsonr(Y_test, y_test_pred)

    # Plot Each Mouse-Brain Area-Sound Type Combo with Predicted and True Frequencies
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=Y_test_sorted, y=y_test_pred_sorted, color='black', label="Predicted vs True")
    sns.regplot(x=Y_test_sorted, y=y_test_pred_sorted, scatter=False, color='red',
                line_kws={"linestyle": "--", "linewidth": 2}, label="Regression Line")

    plt.xlabel('True Frequency (Sorted)')
    plt.ylabel('Predicted Frequency')
    plt.title(f"Predicted vs True Frequency for {key[0]} - {key[1]} - {key[2]}\nBest Alpha: {best_alpha}")
    plt.text(0.5, 0.9, f"R²: {best_r2:.3f}, RMSE: {rmse:.3f}\nNeurons: {n_neurons}\nPearson's r: {corr:.3f}",
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='blue')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure to the specified directory
    plt.savefig(f"/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Ridge Regression/{key[0]}/"
                f"{key[1]} ({key[2]}) {key[0]} - True vs Predicted")
    plt.close()
    #plt.show()

# Plot for n_neurons v. test R^2
plt.figure(figsize=(12, 12))
sns.scatterplot(x=n_neurons_list, y=r2_test_list, hue=labels_list, palette='viridis', s=100)

plt.title("Test $R^2$ vs. Number of Neurons", fontsize=16)
plt.xlabel("Number of Neurons (n_neurons)", fontsize=14)
plt.ylabel("Test $R^2$", fontsize=14)
plt.legend(title="Mouse - Brain Area - Sound Type", bbox_to_anchor=(0.8, 0.5), loc='center left', fontsize=6)
plt.grid(True)
plt.tight_layout()

plt.savefig("/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Ridge Regression/R2_vs_Neurons.png")
plt.show()