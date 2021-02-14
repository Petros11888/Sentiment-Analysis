import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier



Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'z': [1,1,2,2,3,3,3,2,3,3,7,5,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
  
df = DataFrame(Data,columns=['x','z','y'])
print(df)

#centroids = kmeans.cluster_centers_
#print(centroids)

#plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
#plt.show()
features = df.iloc[:, 0].values
labels = df.iloc[:, 2].values
X_train, X_test, y_train, y_test= train_test_split(features,labels,test_size=0.2, random_state=0)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_train, y_train)
y_kmeans = kmeans.predict(df)

print(confusion_matrix(y_test,y_kmeans))
print(classification_report(y_test,y_kmeans))
print(accuracy_score(y_test, y_kmeans))
#plt.show()