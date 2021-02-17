import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas import DataFrame
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import re

df = pd.read_csv("pos3.csv" )
print(df.head())
sentences = []
length = []
stop_words = set(stopwords.words('english'))
i = 0
lmtzr = WordNetLemmatizer()
while i <= 1:
    wrd = df.iloc[i,1]
    t1 = re.sub(r'http\S+', '', wrd)
    token = nltk.word_tokenize(t1)
    token = [word.lower() for word in token if word.isalpha()]
    filtered_words = [word for word in token if word not in stopwords.words('english')]
    lemmatized = [[lmtzr.lemmatize(word) for word in word_tokenize(s)]
              for s in filtered_words]
    sentences.append(lemmatized)
    length.append(len(lemmatized))                
    i += 1
  
  
    
print(sentences)    
print(length)
 
