import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas import DataFrame
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

df = pd.read_csv("all_data.csv", )

sentences = df.iloc[:, 1].values
length = []
together=[]
separator = ' '
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()
for i in range(0,len(sentences)):
    t = sentences[i]
    t = re.sub(r'\s+[a-zA-Z]\s+', ' ', t)
    t = re.sub(r'http\S+', '', t)
    t = re.sub(r'@\S+', '', t)
    t = re.sub(r'#\S+', '', t)
    t = re.sub(r'\s+', ' ', t, flags=re.I)
    token_list = nltk.word_tokenize(t)
    token_list = [word.lower() for word in token_list if word.isalpha()]
    token_list = [word for word in token_list if word not in stopwords.words('english')]
    token_list = [lmtzr.lemmatize(token) for token in token_list]
    #token_list = [stemmer.stem(token) for token in token_list]       
    together.append(separator.join(token_list))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(together)

y = df.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.20)

svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='rbf')
#svclassifier = SVC(kernel='poly', degree=3)
#svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))