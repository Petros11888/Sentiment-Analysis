from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas import DataFrame
import string
import nltk


df = pd.read_csv("pos3.csv" )
b = range(0,2)
a = []
c=[]
for i in b:
    wrd = df.iloc[i,1]
    wrd_no_punctuation = wrd.translate(str.maketrans('', '', string.punctuation))#bgazei ta shmeia stikshs
    tokenlist = nltk.word_tokenize(wrd_no_punctuation)
    a.append(tokenlist)
    c.append(len(tokenlist))

print(a)   
print(c)
