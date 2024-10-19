#%% Imports
import numpy as np
from jaratoolbox import extraplots
import matplotlib.pyplot as plt
from jaratoolbox import colorpalette as colp
import matplotlib.patches as patches


#%% Code
fontSizeLabels = 10
xTicks = [-0.03, 0.1]

# Create the decay segment (sigmoidal)
def sigmoid(x, a, b):
    return a / (1 + np.exp(b * x))


peakFR = [40, 60, 30, 20]
time_range = np.arange(-0.05, 0.2, 0.001)

# Define the time points (250 in total)
time_points = np.arange(250)

colors = [colp.TangoPalette['ScarletRed1'], colp.TangoPalette['ScarletRed2'],
          colp.TangoPalette['Chameleon1'], colp.TangoPalette['Chameleon2']]

for index, peak in enumerate(peakFR):
    spike_counts = np.random.randint(2, 6, 250)
    time_decay = np.linspace(0, 1, 50)  # 100 time points for the decay
    decay_segment = peak * sigmoid(time_decay, 1, 5)  # Adjust parameters for desired shape

    # Insert the increase and decay segments into the spike_counts array
    spike_counts[50] = peak
    spike_counts[51:101] = decay_segment

    # Plotting
    spike_counts = spike_counts.reshape(1, 250)
    psth = extraplots.plot_psth(spike_counts, 6, time_range,
                                colorEachCond=[colors[index]])

    plt.ylabel('Firing rate (spk/s)', fontsize=fontSizeLabels, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=fontSizeLabels, fontweight='bold')
    plt.xticks(xTicks)
    plt.xlim([-0.03, 0.1])

    if index == 1:
        plt.legend(["Pure Tone - 15 kHz", "Pure Tone - 25 kHz"])
        extraplots.save_figure(f"psth_selectivity_{index}", "svg", [6, 4],
                           "/Users/Matt/Desktop/Research/Murray/data/images/proposal_images/")
        plt.show()

    elif index == 3:
        plt.legend(["AM - 16 Hz", "AM - 32 Hz"])
        extraplots.save_figure(f"psth_selectivity_{index}", "svg", [6, 4],
                           "/Users/Matt/Desktop/Research/Murray/data/images/proposal_images/")
        plt.show()

#%% Single psth with evoked range highlighted

psth = extraplots.plot_psth(spike_counts, 6, time_range)
yLims = np.array(plt.ylim())
ax = plt.gca()

plt.axvline(x=0, ls='--')
soundPatch = patches.Rectangle((0.0, yLims[1]*1.03), 0.05, yLims[1]*0.04,
                               linewidth=1, edgecolor="k",
                               facecolor="k", clip_on=False)
ax.add_patch(soundPatch)

plt.ylabel('Firing rate (spk/s)', fontsize=fontSizeLabels, fontweight='bold')
plt.xlabel('Time (s)', fontsize=fontSizeLabels, fontweight='bold')
plt.xticks(xTicks)

extraplots.save_figure("psth_single_cell_trial", "svg", [6, 4],
                       "/Users/Matt/Desktop/Research/Murray/data/images/proposal_images/")

plt.show()

#%% Raster?
spikes_plot = spikeTimesFromEventOnsetAll[1]
index_plot = indexLimitsEachTrialAll[1]

rasterp, hline, zline = extraplots.raster_plot(spikes_plot, index_plot, [-0.2, 0.25])

yLims = np.array(plt.ylim())

ax = rasterp[0].axes

ax.annotate('Evoked Response', xy=(0.51, 1.01), xytext=(0.51, 1.05),
            fontsize=12, fontweight='bold', ha='center',
            va='bottom', xycoords='axes fraction',
            bbox=dict(boxstyle='square', fc='0.8'),
            arrowprops=dict(arrowstyle='-[, widthB=1.3, lengthB=0.5', lw=2.0, color='k'))

# ax.annotate('Evoked Response', xy=(0.03, yLims[1]), xytext=(0.03, yLims[1]*1.05),
#             fontsize=fontSizeLabels, fontweight='bold', ha='center',
#             va='bottom', xycoords='data',
#             bbox=dict(boxstyle='square', fc='0.8'),
#             arrowprops=dict(arrowstyle='-[, widthB=0.9, lengthB=.5', lw=2.0))

# Label axes
plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
plt.ylabel('Trial Index', fontsize=12, fontweight='bold')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

extraplots.save_figure("raster_single_cell_trial", "svg", [6, 4.1],
                       "/Users/Matt/Desktop/Research/Murray/data/images/proposal_images/")


plt.show()
