import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pandas import DataFrame
import nltk

df = pd.read_csv("income.csv")
#print(df.head())

X = df.iloc[:,1]
y = df.iloc[:,2]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')

classifier.fit(X_train, y_train)
X_test = X_test.reshape(-1,1)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
