import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas import DataFrame
import nltk
from nltk.corpus import stopwords 
 


df = pd.read_csv("pos3.csv" )
print(df.head())
b = range(0,2)
sentences = []
length = []
filtered = []
sentiments = []
stop_words = set(stopwords.words('english'))

i = 0
while i < 2:
    wrd = df.iloc[i,1]
    token = nltk.word_tokenize(wrd)
    for w in token:
        if w not in stop_words:
            filtered.append(w)
    sentences.append(filtered)
    length.append(len(filtered))                
    i += 1
    
print(sentences)    
print(length)
 

