#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from collections import Counter


# In[2]:


train_data = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.train.txt.5k",header = None, 
                         names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                                'hours-per-week', 'country-of-origin', 'target'])


# In[3]:


dev_set = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.dev.txt",header = None, 
                      names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                             'hours-per-week', 'country-of-origin', 'target'])


# In[4]:


# Train_Data
labels = train_data['target']


# Dev_set
d = dev_set['target']


# In[5]:


train_features = train_data.drop(['target'], axis = 1)


# In[6]:


num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[7]:


preprocessor = ColumnTransformer([
('num', num_processor, ['age','hours-per-week']),
('cat', cat_processor, ['sector','education', 'marital-status', 'occupation', 'race', 'sex','country-of-origin'])
])
preprocessor.fit(train_data)
train_processed_data = preprocessor.transform(train_data)
dev_processed_data = preprocessor.transform(dev_set)


# In[8]:


# Train_data
X_train= train_processed_data
y_train= labels

# Dev_data
X_dev =  dev_processed_data
y_dev = d


# In[9]:


# Choosing the first person in the dev dataset
query_person_features = X_dev[0].reshape(1, -1)


# In[10]:


# EUCLIDEAN K-NN ALGORITHM



k  = 3
# Creating and fitting the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=k, metric = 'euclidean')
knn.fit(X_train,y_train)
    
# Finding the three closest individuals from the training set to the query person
distances, indices = knn.kneighbors(query_person_features)
    
y_train_pred = knn.predict(X_train)
y_dev_pred = knn.predict(X_dev)
    
train_error_rate = 1 - accuracy_score(y_train, y_train_pred)
dev_error_rate =  1 - accuracy_score(y_dev, y_dev_pred)
   
# Extracting the indices and distance of the closest individuals
top3_indices = indices[0]
top3_distances = distances[0]

# Extracting the features of the top-3 closest individuals from the training dataset
top3_individuals_features = y_train[top3_indices]
    
# Print the indices of the closest individuals in the training set
print(f"dev error: {   dev_error_rate * 100:.2f}")
print(f"train error: {   train_error_rate * 100:.2f}")
print(f"query person: {query_person_features}")
print("Indices of the top-3 closest individuals in the training set:")
print(f"k = {k}",end='\t')
print(f"top3_indices :{top3_indices}", end='\t')
print(f"top3_distances :{ top3_distances}", end='\t')
print(f"top3_data points :{ top3_individuals_features}")


# In[ ]:




