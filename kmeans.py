from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('ClusterPlot.csv', header=0)
df = df[['V1', 'V2']]

silhouette_scores = []

for n_clusters in range(2, 8):
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++')
    cluster_labels = clusterer.fit_predict(df)

    silhouette_avg = silhouette_score(df, cluster_labels)
    silhouette_scores.append((n_clusters, silhouette_avg))

max_score = max(silhouette_scores, key=lambda i: i[1])
n_clusters = max_score[0]
print('There are', n_clusters, 'clusters.')

clusterer = KMeans(n_clusters=n_clusters, init='k-means++')
cluster_labels = clusterer.fit_predict(df)

g = sns.relplot(x='V1', y='V2', hue=cluster_labels, data=df, legend='full')
legend = g._legend
legend.set_title('nth cluster')

plt.show()
