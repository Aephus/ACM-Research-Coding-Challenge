from sklearn.cluster import DBSCAN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('ClusterPlot.csv', header=0)
df = df[['V1', 'V2']]
x = df['V1']
y = df['V2']

plt.plot(x, y, 'ro')

db = DBSCAN(min_samples=3).fit(df)
cluster_labels = db.labels_

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
df['cluster_labels'] = cluster_labels
print(cluster_labels)

plot = sns.relplot(x='V1', y='V2', hue=cluster_labels, data=df)
#plot.legend(title='n cluster')
plt.show()
