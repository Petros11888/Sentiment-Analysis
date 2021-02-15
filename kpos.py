import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas import DataFrame
import string
import nltk


df = pd.read_csv("pos3.csv" )
print(df.head())
b = range(0,2)
a = []
for i in b:
    text = df.iloc[i,1]
    text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation))#bgazei ta shmeia stikshs
    tokenlist = nltk.word_tokenize(text_no_punctuation)
    a.append(tokenlist)
    
print(a)    
    
