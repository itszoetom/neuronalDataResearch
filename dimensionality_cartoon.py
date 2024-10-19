#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from jaratoolbox import extraplots
from sklearn.decomposition import PCA


#%% FR cloud generation
np.random.seed(1)
fontSizeLabels = 12

num_points = 200

cloud = np.linspace(6, 20, num_points) + np.random.normal(0, 1, num_points)
corresponding = 1.5 * cloud + np.random.normal(0, 5, num_points)
cloud = np.stack([cloud, corresponding], axis=1).flatten().reshape((200, 2))

#%%
responsePCA = PCA()
PC_resp = responsePCA.fit(cloud)
resp_eigenvectors = responsePCA.components_

plt.scatter(cloud[:, 0], cloud[:, 1], color="c", alpha=0.5)
ax = plt.gca()
ax.quiver(12.5, 20, *resp_eigenvectors[:, 0], color='r',
          scale=7)
ax.quiver(12.5, 20, *resp_eigenvectors[:, 1], color='k',
          scale=11)

plt.ylabel('Firing rate cell 2 (spk/s)', fontsize=fontSizeLabels, fontweight='bold')
plt.xlabel('Firing rate cell 1 (spk/s)', fontsize=fontSizeLabels, fontweight='bold')

plt.legend(["Trial response", "PC 1", "PC 2"], loc="best")

extraplots.save_figure("pca_cloud", "svg", [6, 4],
                       "/Users/Matt/Desktop/Research/Murray/data/images/proposal_images/")

plt.show()

# Transformed data
# trans_cloud = responsePCA.transform(cloud)
#
# plt.scatter(trans_cloud[:, 0], trans_cloud[:, 1])
#
# plt.show()
