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
    token = nltk.word_tokenize(t2)
    token = [word.lower() for word in token if word.isalpha()]
    filtered_words = [word for word in token if word not in stopwords.words('english')]
    for s in filtered_words:
        lmtzr.lemmatize(s)
    sentences.append(filtered_words)
    
    together.append(separator.join(filtered_words))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(together)
#print(X.toarray())      
#print(df.head())
y = df.iloc[:, 2].values

# print(X)
print(y)


X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=20)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))