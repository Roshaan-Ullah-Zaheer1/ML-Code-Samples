import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

.
# load data from CSV file, use first row as column names
data = pd.read_csv('D:/Semester 6/Machine Learning/Assignment 4/housing_data.csv', header=0)

# select longitude and latitude columns
coords = data[['longitude', 'latitude']]

# initialize KMeans with 6 clusters
kmeans = KMeans(n_clusters=6)

# fit KMeans to data
kmeans.fit(coords)

# get cluster labels for each data point
labels = kmeans.labels_

# manually set color for each cluster
colors = ['blue', 'orange', 'green', 'pink', 'red', 'yellow']

# plot the scatter plot with seaborn
plt.figure(figsize=(6, 6))
sns.scatterplot(x='longitude', y='latitude', data=coords, hue=labels, palette=colors, s=20, alpha=1)

# add legend with diamond markers and numbering
legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=format(i), markersize=10, linestyle='') for i, color in enumerate(colors)]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), title='Cluster')

plt.title("Housing Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)

plt.show()
