import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pandas import DataFrame
import nltk

df = pd.read_csv("neg5.csv")
print(df.head())

wrd = df.iloc[1,1]

token = nltk.word_tokenize(wrd)
print(token)
