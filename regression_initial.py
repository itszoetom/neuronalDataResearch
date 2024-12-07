import numpy as np
import matplotlib.pyplot as plt
import os
from jaratoolbox import celldatabase, settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import funcs

# %% Constants
subject_list = ['feat005', 'feat006', 'feat007', 'feat008', 'feat009'] # 'feat004', 'feat010'
recordingDate_list = {
    'feat004': ['2022-01-11', '2022-01-19', '2022-01-21'],
    'feat005': ['2022-02-07', '2022-02-08', '2022-02-11', '2022-02-14', '2022-02-15', '2022-02-16'],
    'feat006': ['2022-02-21', '2022-02-22', '2022-02-24', '2022-02-25', '2022-02-26', '2022-02-28', '2022-03-01',
                '2022-03-02'],
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

# %% Load dataframe
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
fullDb["recordingSiteName"] = simpleSiteNames

X_speech_array, X_AM_array, X_PT_array,\
    Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all, \
    Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted \
    = funcs.clean_and_concatenate(subject_list, recordingDate_list, targetSiteNames,
                                                                     previous_frequency_AM, previous_frequency_PT,
                                                                     previous_frequency_speech)

# Add data to the dictionary for each brain area and sound type
for subject in subject_list:
    for brain_area in ["Primary auditory area", "Ventral auditory area"]:  # Removed speech sounds for now
        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['AM', 'PT'],
                [X_AM_array, X_PT_array],
                [Y_brain_area_AM_all, Y_brain_area_PT_all],
                [Y_frequency_AM_sorted, Y_frequency_pureTones_sorted]):
            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]
            X_array_adjusted = X_array_adjusted.T
            data_dict[(subject, brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

# Define the alpha range with a wider spectrum for potentially more variation
alphas = np.logspace(-2, 3, 100)  # Extend lambda values

# Define tolerance level for "percent correct" evaluation
tolerance = 0.05  # 5% tolerance of the target range

for key, value in data_dict.items():
    X = value['X']
    Y = value['Y']

    # Standardize X to improve Ridge performance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Store R-squared, MSE, and percent correct for each alpha
    train_acc = np.empty_like(alphas)
    test_acc = np.empty_like(alphas)
    train_mse = np.empty_like(alphas)
    test_mse = np.empty_like(alphas)
    train_percent_correct = np.empty_like(alphas)
    test_percent_correct = np.empty_like(alphas)

    for i, alpha in enumerate(alphas):
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train, Y_train)

        # R-squared values
        train_acc[i] = ridge_reg.score(X_train, Y_train)
        test_acc[i] = ridge_reg.score(X_test, Y_test)

        # Mean Squared Error (MSE) values
        train_mse[i] = mean_squared_error(Y_train, ridge_reg.predict(X_train))
        test_mse[i] = mean_squared_error(Y_test, ridge_reg.predict(X_test))

        # Percent correct
        y_train_pred = ridge_reg.predict(X_train)
        y_test_pred = ridge_reg.predict(X_test)
        train_range = Y_train.max() - Y_train.min()
        test_range = Y_test.max() - Y_test.min()

        train_percent_correct[i] = np.mean(np.abs(Y_train - y_train_pred) <= tolerance * train_range) * 100
        test_percent_correct[i] = np.mean(np.abs(Y_test - y_test_pred) <= tolerance * test_range) * 100

    # Identify the best alpha with the highest R-squared on the test set
    best_alpha_idx = np.argmax(test_acc)
    best_alpha = alphas[best_alpha_idx]
    best_train_mse = train_mse[best_alpha_idx]
    best_test_mse = test_mse[best_alpha_idx]
    best_train_r2 = train_acc[best_alpha_idx]
    best_test_r2 = test_acc[best_alpha_idx]
    best_train_pc = train_percent_correct[best_alpha_idx]
    best_test_pc = test_percent_correct[best_alpha_idx]

    # Print the results
    print(f"Model for {key[0]} (brain area: {key[1]}, sound type: {key[2]}) - Best Lambda: {best_alpha}")
    print(f"Train MSE: {best_train_mse}, Test MSE: {best_test_mse}")
    print(f"Train R-squared: {best_train_r2}, Test R-squared: {best_test_r2}")
    print(f"Train Percent Correct: {best_train_pc}%, Test Percent Correct: {best_test_pc}%")
    print()  # For better readability between results

    # Plotting R-squared, MSE, and Percent Correct values
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(alphas, train_acc, label="Train R-squared", color='blue')
    axs[0].plot(alphas, test_acc, label="Test R-squared", color='red')
    axs[0].axhline(1 / len(np.unique(Y)), color='gray', linestyle='--', label='Baseline')
    axs[0].set_xscale('log')
    axs[0].set_xlabel("Lambda (α)")
    axs[0].set_ylabel("R-squared")
    axs[0].legend()
    axs[0].set_title(f"R-squared vs Lambda for {key}")

    axs[1].plot(alphas, train_mse, label="Train MSE", color='blue')
    axs[1].plot(alphas, test_mse, label="Test MSE", color='red')
    axs[1].set_xscale('log')
    axs[1].set_xlabel("Lambda (α)")
    axs[1].set_ylabel("Mean Squared Error (MSE)")
    axs[1].legend()
    axs[1].set_title(f"MSE vs Lambda for {key}")

    axs[2].plot(alphas, train_percent_correct, label="Train Percent Correct", color='blue')
    axs[2].plot(alphas, test_percent_correct, label="Test Percent Correct", color='red')
    axs[2].set_xscale('log')
    axs[2].set_xlabel("Lambda (α)")
    axs[2].set_ylabel("Percent Correct (%)")
    axs[2].legend()
    axs[2].set_title(f"Percent Correct vs Lambda for {key}")

    plt.tight_layout()
    plt.show()