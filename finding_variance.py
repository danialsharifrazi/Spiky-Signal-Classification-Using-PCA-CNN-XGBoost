import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# read data
import read_data
dpi=0
X,labels=read_data.read_all_data(dpi)

# Perform PCA
pca = PCA()
pca.fit(X)

# Calculate cumulative explained variance ratio
cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot
plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.show()


