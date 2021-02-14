from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd  
    
# List1  
lst = [['tom', 25,"apple cool"], ['krish', 30, "banana cool"], 
       ['nick', 26, "banana cool"], ['juli', 22, "banana cool"]] 
    
df = pd.DataFrame(lst, columns =['Name', 'Age','Wrd']) 
print(df)

mylist = ["apple cool cool cool cool", "banana cool", "cherry is cool"]
for item in range(0,5):
    mylist.append("Hi this is")
print(mylist)
vectorizer = TfidfVectorizer (stop_words=stopwords.words('english'))
mylist = vectorizer.fit_transform(mylist).toarray()
print(mylist)