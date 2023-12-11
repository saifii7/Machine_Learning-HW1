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


# In[12]:


# Choosing the first person in the dev dataset
query_person_features = X_dev[0].reshape(1, -1)


# In[13]:


# SELF MNAHATTAN K-NN ALGORITHM


query_point = X_dev[0]
def custom_manhattan_distance(a, b):
#Calculating the Manhattan distance between two data points 'a' and 'b'
    distance = np.sum(np.abs(a - b))
    return distance

distances = [custom_manhattan_distance(query_point,x) for x in X_train]



k = 3


nearest_indices = np.argsort(distances)[:k]
distances.sort()

nearest_neighbors = y_train[nearest_indices]

print("Distances of the three nearest neighbors:", distances [:k])
print("Indices of the three nearest neighbors:", nearest_indices)
print(train_features.iloc[nearest_indices])
print("Nearest neighbor data points:", nearest_neighbors)


# In[ ]:




