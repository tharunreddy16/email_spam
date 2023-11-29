#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv('spam.csv', encoding="ISO-8859-1")
df = df[['v1', 'v2']]
df.columns = ['Category', 'Message']
encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Category, test_size=0.2)


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X_train_count = vec.fit_transform(X_train.values)
X_test_count = vec.transform(X_test)


# In[5]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_count, y_train)


# In[7]:


accuracy = model.score(X_test_count, y_test)
print(f'Accuracy of the model: {accuracy * 100:.2f}%')


# In[8]:


def predict():
    message = input('Enter the message to predict: ')
    message = [message]
    vector = vec.transform(message)
    if model.predict(vector) == 1:
        print("SPAM EMAIL")
    else:
        print("NOT SPAM")

predict()


# In[ ]:





# In[ ]:




