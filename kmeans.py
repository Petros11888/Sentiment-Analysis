#math operations over arrays
import numpy as np

#plotting library
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.datasets import make_blobs

#make_blobs(*args, n_samples=100,centers=None, cluster_std=1.0, random_state=None)
#X : array of shape [n_samples, n_features]. The generated samples.  
#y : array of shape [n_samples]. The integer labels for cluster membership of each sample. 
#cluster_std: Standard Deviation of the clusters
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#Scatter: plots are used to plot data points on a horizontal and a vertical axis 
#in the attempt to show how much one variable is affected by another
#Χ[:,0],Χ[:,1] : Ola ta samples ws pros to label 0 kai ws pros to 1
#s:size of dots
plt.scatter(X[:, 0], X[:, 1], s=50)

#import kMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)

#compute kMeans clustering
kmeans.fit(X)

#Testarei to sunolo X kai epistrefei sto y_kmeans tis provlepseis, provlepei dhladh thn klash
y_kmeans = kmeans.predict(X)

#Scatter: plots are used to plot data points on a horizontal and a vertical axis 
#in the attempt to show how much one variable is affected by another
#Χ[:,0],Χ[:,1] : Ola ta samples ws pros to feature 0 kai ws pros to 1
#c:to classification analoga me thn klash? , s:size of dots, cmap: coloring
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

#returns the coordinates of cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()