from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

df = pd.read_csv('ClusterPlot.csv', header=0)
df = df[['V1', 'V2']]

# plt.plot(x, y, 'ro')
# plt.show()
silhouette_scores = []

for n_clusters in range(2, 8):
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++')
    cluster_labels = clusterer.fit_predict(df)

    silhouette_avg = silhouette_score(df, cluster_labels)
    silhouette_scores.append((n_clusters, silhouette_avg))

max_score = max(silhouette_scores, key=lambda i: i[1])
n_clusters = max_score[0]
print(n_clusters)

clusterer = KMeans(n_clusters=n_clusters, init='k-means++')
cluster_labels = clusterer.fit_predict(df)

plot = sns.relplot(x='V1', y='V2', hue=cluster_labels, data=df)
#plot.legend(title='n cluster')
plt.show()
