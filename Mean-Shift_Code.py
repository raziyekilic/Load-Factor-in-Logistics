import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn import preprocessing
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth


# Read the Excel file
df = pd.read_excel('dataset.xlsx')

# new9
#df2 = df[["Outbound shipment total tonnage", "Outbound tonnage rate", "Outbound pallet rate", "Number of outbound pallets",	"Vehicle type",	"Vehicle departure location",	"Vehicle destination location",	"Vehicle direction",	"Number of inbound pallets",	"Inbound pallet rate",	"Inbound tonnage rate",	"Inbound shipment total tonnage",	"Total waiting time",	"Number of pending shipments",	"Number of waiting shipments",	"Day difference",	"Day difference2",	
#"Waiting time performance coefficient",	"The cost coefficient",	"The maximum occupancy coefficient",	"The capacity factor",	"The shipment coefficient"]]

# new10
df2 = df[["Outbound tonnage rate", "Outbound pallet rate", "Inbound pallet rate",	"Inbound tonnage rate", "Waiting time performance coefficient", "The cost coefficient",	"The maximum occupancy coefficient",	"The capacity factor",	"The shipment coefficient"]]

# new11
# df2 = df[["Waiting time performance coefficient", "The cost coefficient", "The maximum occupancy coefficient", "The capacity factor", "The shipment coefficient"]]

# Normalize the data
df2 = preprocessing.normalize(df2)

# bandwidth = estimate_bandwidth(df2, quantile=0.3, n_samples=2000) #new9

bandwidth = estimate_bandwidth(df2, quantile=0.15, n_samples=500) #new10

# bandwidth = estimate_bandwidth(df2, quantile=0.2, n_samples=500) #new11


ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(df2)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()

colors = ["#dede00", "#377eb8", "#f781bf"]
markers = ["x", "o", "^"]

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(df2[my_members, 1], df2[my_members, 2], markers[k], color=col)
    plt.plot(
        cluster_center[1],
        cluster_center[2],
        markers[k],
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()


sutun_name='mean_shift:'+str(n_clusters_)
df[sutun_name] = labels


plt.show()

df.to_excel("new3.xlsx")

