import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from jaratoolbox import celldatabase, settings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import funcs
import seaborn as sns

# %% Constants
subject_list = ['feat005', 'feat006', 'feat007', 'feat008', 'feat009']  # 'feat004', 'feat010'
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
binWidth = 0.01
binEdges = np.arange(evoked_start, evoked_end, binWidth)
periodsNameSpeech = ['base200', 'respOnset', 'respSustained']
allPeriodsSpeech = [[-0.2, 0], [0, 0.12], [0.12, 0.24]]
timeRangeSpeech = [allPeriodsSpeech[0][0], allPeriodsSpeech[-1][-1]]
binSize = 0.005
binEdgesSpeech = np.arange(allPeriodsSpeech[1][0], allPeriodsSpeech[1][1], binSize)
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

# %% Load dataframe
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
fullDb["recordingSiteName"] = simpleSiteNames

# Add data to the dictionary for each brain area and sound type
for subject in subject_list:
    for brain_area in ["Primary auditory area", "Ventral auditory area"]:  # Removed speech sounds for now

        X_speech_array, X_AM_array, X_PT_array, \
            Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all, \
            Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted \
            = funcs.clean_and_concatenate(subject, recordingDate_list, brain_area,
                                          previous_frequency_AM, previous_frequency_PT,
                                          previous_frequency_speech)

        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['AM', 'PT'],
                [X_AM_array, X_PT_array],
                [Y_brain_area_AM_all, Y_brain_area_PT_all],
                [Y_frequency_AM_sorted, Y_frequency_pureTones_sorted]):

            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]
            X_array_adjusted = X_array_adjusted.T
            data_dict[(subject, brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

# 5% tolerance of the target range
tolerance = 0.05
# Define the alpha range
alphas = np.logspace(-5, 5, 200)

for key, value in data_dict.items():
    X = value['X']
    Y = np.log(value['Y'])
    n_neurons = X.shape[1]
    n_neurons_list.append(n_neurons)
    if key[1] == 'Primary auditory area':
        primary_n_neurons += n_neurons
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

    # Save the best R^2 value for this combo
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

    # Plot Each Mouse-Brain Area-Sound Type Combo with Predicted and Test Points
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=np.arange(len(Y_test_sorted)), y=Y_test_sorted, color='black', label="Sorted Test Points")
    sns.scatterplot(x=np.arange(len(y_test_pred_sorted)), y=y_test_pred_sorted, color='red', label="Sorted Predicted Points")

    plt.xlabel('Trials')
    plt.ylabel('Frequency')
    plt.title(f"Sorted Predicted vs Test Points for {key[0]} - {key[1]} - {key[2]}\nBest Alpha: {best_alpha}")
    plt.text(0.5, 0.9, f"R²: {best_r2:.3f}", ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='blue')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    # Save the figure to the specified directory
    plt.savefig(f"/Users/zoetomlinson/Desktop/NeuroAI/Figures/Model Plots/{key[0]}/"
                f"{key[1]} ({key[2]}) Ridge Regression Plot for {key[0]}")
    plt.show()

# Plot for n_neurons v. test R^2
plt.figure(figsize=(10, 6))
sns.scatterplot(x=n_neurons_list, y=r2_test_list, hue=labels_list, palette='viridis', s=100)

plt.title("Test $R^2$ vs. Number of Neurons", fontsize=16)
plt.xlabel("Number of Neurons (n_neurons)", fontsize=14)
plt.ylabel("Test $R^2$", fontsize=14)
plt.legend(title="Mouse - Brain Area - Sound Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.savefig("/Users/zoetomlinson/Desktop/NeuroAI/Figures/Model Plots/R2_vs_Neurons.png")
plt.show()

# Plot Box Plot of All Brain Area-Sound Type R^2
plot_data = pd.DataFrame({
    'R2 Test': r2_test_list,
    'Neurons': n_neurons_list,
    'Label': labels_list})
plot_data[['Subject', 'Brain Area', 'Sound Type']] = plot_data['Label'].str.split(' - ', expand=True)

plt.figure(figsize=(12, 8))
sns.boxplot(x='Brain Area', y='R2 Test', hue='Sound Type', data=plot_data, palette='Set2', dodge=True)

# Overlay individual points for each subject
sns.stripplot(x='Brain Area', y='R2 Test', hue='Sound Type',
              data=plot_data, palette='dark:.3', dodge=True, jitter=True,
              marker='o', alpha=0.7, linewidth=0.5)

handles, labels = plt.gca().get_legend_handles_labels()
l = plt.legend(handles[0:len(unique_labels)], labels[0:len(unique_labels)], loc='upper right', title='Sound Type')

plt.title("R2 Test Values for Brain Area and Sound Type", fontsize=16)
plt.xlabel("Brain Area", fontsize=14)
plt.ylabel("R2 Test Value", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show the plot
plt.savefig("/Users/zoetomlinson/Desktop/NeuroAI/Figures/Model Plots/Boxplot_BrainArea_SoundType.png")
plt.show()



'''
# Plot Train and Test Points with Connecting Line
    # Add a jitter for better visualization of overlapping points
    jitter = 0.02 * np.random.randn(len(Y_test))

    # Create the new plot
    plt.figure(figsize=(10, 6))
    for i in range(len(Y_test)):
        plt.plot([0, 1], [Y_test[i], y_test_pred[i]], color='gray', alpha=0.5)  # Line connecting test to pred

    # Scatter the test points
    sns.scatterplot(x=np.zeros(len(Y_test)) + jitter, y=Y_test, color='blue', label="Test Points", alpha=0.8)
    # Scatter the predicted points
    sns.scatterplot(x=np.ones(len(y_test_pred)) + jitter, y=y_test_pred, color='orange', label="Predicted Points", alpha=0.8)

    # Customize the plot
    plt.xticks([0, 1], ['Test', 'Pred'])
    plt.xlabel('Data Points')
    plt.ylabel('Frequency')
    plt.title(f"Test vs Prediction Frequencies - {key[1]} ({sound_type}) - {key[0]}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    # Save the figure to the specified directory
    plt.savefig(f"/Users/zoetomlinson/Desktop/NeuroAI/Figures/Model Plots/"
                f"{key[2]} ({key[1]}) Test vs Pred Connection Plot for {key[0]}")
    plt.show()'''
