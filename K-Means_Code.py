import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn import preprocessing
import pandas as pd

# Read the Excel file
df = pd.read_excel('dataset.xlsx')

# new9
#df2 = df[["Outbound shipment total tonnage", "Outbound tonnage rate", "Outbound pallet rate", "Number of outbound pallets",	"Vehicle type",	"Vehicle departure location",	"Vehicle destination location",	"Vehicle direction",	"Number of inbound pallets",	"Inbound pallet rate",	"Inbound tonnage rate",	"Inbound shipment total tonnage",	"Total waiting time",	"Number of pending shipments",	"Number of waiting shipments",	"Day difference",	"Day difference2",	
#"Waiting time performance coefficient",	"The cost coefficient",	"The maximum occupancy coefficient",	"The capacity factor",	"The shipment coefficient"]]

# new10
df2 = df[["Outbound tonnage rate", "Outbound pallet rate", "Inbound pallet rate",	"Inbound tonnage rate", "Waiting time performance coefficient", "The cost coefficient",	"The maximum occupancy coefficient",	"The capacity factor",	"The shipment coefficient"]]

# new11
#df2 = df[["Waiting time performance coefficient", "The cost coefficient", "The maximum occupancy coefficient", "The capacity factor", "The shipment coefficient"]]

# Normalize the data
df2 = preprocessing.normalize(df2)

# Clustering algorithms and parameters
clustering_algorithms = {
    "K-Means": KMeans,
}
n_clusters_list = [3]
random_state = 0

# Create subplots
fig, axs = plt.subplots(
    nrows=len(clustering_algorithms),
    ncols=len(n_clusters_list),
    figsize=(12, 5)
)

# Check the structure of axs and process accordingly
if isinstance(axs, np.ndarray):
    axs = axs.T  # Transpose if axs is a NumPy array
else:
    axs = np.array([[axs]])  # Convert to array if single axis

# Run the model for each algorithm and number of clusters
for i, (algorithm_name, Algorithm) in enumerate(clustering_algorithms.items()):
    for j, n_clusters in enumerate(n_clusters_list):
        algo = Algorithm(n_clusters=n_clusters, random_state=random_state, n_init=3)
        algo.fit(df2)
        centers = algo.cluster_centers_

        # Plot data points and cluster centers
        axs[j, i].scatter(df2[:, 3], df2[:, 0], s=10, c=algo.labels_)
        axs[j, i].scatter(centers[:, 4], centers[:, 4], c="r", s=20)
        axs[j, i].set_title(f"{algorithm_name} : {n_clusters} clusters")

# Remove axis labels and ticks in subplots
for ax in axs.flat:
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# Perform final clustering with K-Means and save the results
kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=3).fit(df2)
clusters = kmeans.labels_
df["Cluster-3"] = clusters
df.to_excel("dataset2.xlsx")
