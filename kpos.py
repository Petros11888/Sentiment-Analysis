import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas import DataFrame
import nltk


df = pd.read_csv("pos3.csv" )
print(df.head())
b = range(0,2)
a = []
c = []
for i in b:
    wrd = df.iloc[i,1]
    token = nltk.word_tokenize(wrd)
    a.append(token)
    c.append(len(token))
    if i == 2:
        break
    
print(a)    
print(c)    
