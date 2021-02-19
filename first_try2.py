import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas import DataFrame
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.svm import SVC
#names=['id', 'text', 'sentiment']
df = pd.read_csv("all_data.csv", )

#print(df.head())
sentences = []
length = []
together=[]
separator = ' '
stop_words = set(stopwords.words('english'))
i = 0
lmtzr = WordNetLemmatizer()
for i in range(0,len(df.index)):
    wrd = df.iloc[i,1]
    t1 = re.sub(r'http\S+', '', wrd)
    t2 = re.sub(r'@\S+', '', t1)
    token_list = nltk.word_tokenize(t2)
    low_token_list = [word.lower() for word in token_list if word.isalpha()]
    filtered_words = [word for word in low_token_list if word not in stopwords.words('english')]
    lemmatized = [lmtzr.lemmatize(s) for s in filtered_words]     
    together.append(separator.join(lemmatized))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(together)
#print(X.toarray())      
#print(df.head())
#y = df.iloc[:, 2].values

# print(X)
#print(y)
df.shape
df.head()
X = df.drop('sentiment', axis=1)
y = df['sentiment']

#X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#scaler = StandardScaler()
#scaler.fit(X_train)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)


#classifier = KNeighborsClassifier(n_neighbors=32)
#classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))