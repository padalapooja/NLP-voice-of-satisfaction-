#--PROJECT-Voices of Satisfaction
# Natural Language Processing

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importin the dataset(text)
dataset = pd.read_csv(r"C:\Users\POOJA\Downloads\27th (1)\27th\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv", delimiter='\t',quoting=3)

# Cleaning the text
import re
import nltk

#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range (0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the bag of word model
'''from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values'''

# creating the term frequence and inverse document frequency(tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tf= TfidfVectorizer()
x = tf.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into train and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# Training the Navie Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(x_test)


# Training the Logistic model on the Training set
'''
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(x_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(x_test)
'''

# Training KNN model on the training set
'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(x_test)
'''
# Training Decision tree model on the training set
'''from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(x_test)'''

# Training support vector machine model on the training set
'''from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(x_test)'''

# Training Ensemble learning model on the training set
'''from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(x_test)'''


# Making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy score
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# Classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(x_train,y_train)
bias

variance = classifier.score(x_test,y_test)
variance



