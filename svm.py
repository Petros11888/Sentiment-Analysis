#sklearn: Python module, integrated classic ML algorithms
#datasets: Popular reference datasets
#make blobs: Gausian distribution
from sklearn.datasets import make_blobs

#make_blobs(*args, n_samples=100,centers=None, cluster_std=1.0, random_state=None)
#X : array of shape [n_samples, n_features]. The generated samples.  
#y : array of shape [n_samples]. The integer labels for cluster membership of each sample. 
#cluster_std: Standard Deviation of the clusters
Χ,y=make_blobs(n_samples=500,centers=2,cluster_std=0.4, random_state=0)

#matplotlib.pyplot is a state-based interface to matplotlib. 
#It provides a MATLAB-like way of plotting.
import matplotlib.pyplot as plt

#numpy module: array object & math operation over arrays
import numpy as np

#Scatter: plots are used to plot data points on a horizontal and a vertical axis 
#in the attempt to show how much one variable is affected by another
#Χ[:,0],Χ[:,1] : Ola ta samples ws pros to feature 0 kai ws pros to 1
#c:analoga me ton a3ona y , s:size of dots, cmap: coloring
plt.scatter(Χ[:, 0], Χ[:, 1], c=y , s=50, cmap="plasma")

#Orismos tou a3ona x
xfit=np.linspace(-1,3.5)

#Support vector machines are one way to address this. 
# What support vector machined do is to not only draw a line, 
# but consider a region about the line of some given width. 
#y=ax+b
for m, b, d in [(1, 0.65, 0.30), (0.5, 1.6, 0.5), (-0.2, 2.9, 0.3)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit)
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
    color='#AFFEDC', alpha=0.3)
#Set the alpha value used for blending 


#Setting limits turns autoscaling off for the x-axis.
plt.xlim(-1,3.5)

plt.show()