#The module:sklearn.naive_bayes module implements Naive Bayes algorithms. 
# These are supervised learning methods based on applying Bayes' theorem 
# with strong (naive) feature independence assumptions.
#Imports Gauss classs
from sklearn.naive_bayes import GaussianNB
#The module:sklearn.metrics module includes score functions, 
# performance metrics and pairwise metrics and distance computations.
#Compute confusion matrix to evaluate the accuracy of a classification.
from sklearn.metrics import confusion_matrix
#Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
#Naive Bayes classifier for polynomial models
from sklearn import datasets

#Load and return the iris dataset (classification).
#The iris dataset is a classic and very easy multi-class classification dataset.
#3 Samples per class 50, Samples total 150 Dimensionality 4
iris = datasets.load_iris()
#print(iris)

#data = iris.data
#Returns a tuple with each index having the number of corresponding elements.
#shape = data.shape

x=iris.data
y=iris.target

#Split arrays or matrices into random train and test subsets
#test_size : float or int, If float, should be between 0.0 and 1.0 
# and represent the proportion of the dataset to include in the test split. 
#Train 0.7 & Test 0.3 of the sample (150)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#Gaussian Naive Bayes Model
gnb=GaussianNB()

#Predict y for Gaussian Naive Bayes
#Fit Gaussian Naive Bayes according to X, y
#Perform classification on an array of test vectors X.
#Symfwna me to trainf set gnb.fit(x_train,y_train), testarei to test sunolo x_test kai epistrefei sto y_pred_gnb tis provlepseis
y_pred_gnb=gnb.fit(x_train,y_train).predict(x_test)
#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)
print(y_pred_gnb)

#Compute confusion matrix to evaluate the accuracy of a classification.
cnf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)
print(cnf_matrix_gnb)