#!/usr/bin/env python
# coding: utf-8

# # Bharat Intern Data Science
# 
# # SMS Classifier

# # Import Libraries 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


# # Import the Dataset using csv 

# In[2]:


data = pd.read_csv('D:/spam detection.csv',encoding = 'latin-1')


# In[3]:


# It displays few rows of the dataset
data.head()


# In[4]:


# check for missing values in your dataset 
data.isna().sum()


# In[5]:


data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1,inplace=True)


# In[6]:


data.columns = ['Category','Message']


# In[7]:


# It shows first 10 rows in the Dataset
data.head(10)


# In[8]:


data.info()


# In[9]:


data['Category'].value_counts()


# In[10]:


data['Category'].value_counts().plot(kind = 'bar')


# In[11]:


data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)


# In[12]:


data.columns


# In[13]:


# It shows few rows of the Dataset
data.head(10)


# In[14]:


x = np.array(data['Message'])
y = np.array(data['Spam'])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)


# # Predictions

# In[15]:


sample = input('Enter a message')
data = cv.transform([sample]).toarray()
print(clf.predict(data))


# In[17]:


sample = input('Enter a message')
data = cv.transform([sample]).toarray()
print(clf.predict(data))


# # Accuracy

# In[18]:


clf.score(X_test,y_test)

