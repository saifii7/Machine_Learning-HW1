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

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
import math


# In[2]:


train_data = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.train.txt.5k",header = None, 
                         names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                                'hours-per-week', 'country-of-origin', 'target'])


# In[3]:


dev_set = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.dev.txt",header = None, 
                      names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                             'hours-per-week', 'country-of-origin', 'target'])


# In[4]:


# FOR NAIVE BINARIZATION

# Train_Data
n_labels = train_data['target']
n_b = train_data.drop(['target'],axis = 1)

# Dev_set
n_d = dev_set['target']
n_d_features = dev_set.drop(['target'], axis = 1)

# Train Binary Conversion
encoder.fit(n_b)
feature = encoder.transform(n_b)

# Dev Binary conversion
d_f = encoder.transform(n_d_features)

n_X_train= feature
n_y_train= n_labels

n_X_dev = d_f
n_y_dev = n_d



k_values = list(range(1, 101, 2))  

naive_dev_error = []
for k in k_values:  
  
    
    knn = KNeighborsClassifier(n_neighbors=k, metric = "euclidean")
    knn.fit(n_X_train,n_y_train)
   

    n_y_train_pred = knn.predict(n_X_train)
    n_y_dev_pred = knn.predict(n_X_dev)
    
    train_error_rate = 1 - accuracy_score(n_y_train, n_y_train_pred)
    n_dev_error_rate =  1 - accuracy_score(n_y_dev, n_y_dev_pred)
    naive_dev_error.append(n_dev_error_rate)
    
    print(f"k = {k}",end='\t')
    print(f"dev_error = {n_dev_error_rate*100:.2f}")


# In[5]:


# Smart Binarization

# Train_Data
s_labels = train_data['target']


# Dev_set
s_d = dev_set['target']

num_processor = 'passthrough' # i.e., no transformation
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')

preprocessor = ColumnTransformer([
('num', num_processor, ['age','hours-per-week']),
('cat', cat_processor, ['sector','education', 'marital-status', 'occupation', 'race', 'sex','country-of-origin'])
])
preprocessor.fit(train_data)
s_train_processed_data = preprocessor.transform(train_data)
s_dev_processed_data = preprocessor.transform(dev_set)

# Train_data
s_X_train= s_train_processed_data
s_y_train= s_labels

# Dev_data
s_X_dev =  s_dev_processed_data
s_y_dev = s_d


k_values = list(range(1, 101, 2)) 
smart_dev_error = []

for k in k_values:  # Odd values of k
    # Create and fit the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric = 'euclidean')
    knn.fit(s_X_train,s_y_train)
   
    s_y_train_pred = knn.predict(s_X_train)
    s_y_dev_pred = knn.predict(s_X_dev)
    
    s_train_error_rate = 1 - accuracy_score(s_y_train, s_y_train_pred)
    s_dev_error_rate =  1 - accuracy_score(s_y_dev, s_y_dev_pred)
    smart_dev_error.append(s_dev_error_rate)

    print(f"k = {k}",end='\t')
    print(f"dev_error = {s_dev_error_rate*100:.2f}")
   


# In[7]:


# Smart+Smart Binarization

# Train_Data
ss_labels = train_data['target']


# Dev_set
ss_d = dev_set['target']

num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')

preprocessor = ColumnTransformer([
('num', num_processor, ['age','hours-per-week']),
('cat', cat_processor, ['sector','education', 'marital-status', 'occupation', 'race', 'sex','country-of-origin'])
])
preprocessor.fit(train_data)
ss_train_processed_data = preprocessor.transform(train_data)
ss_dev_processed_data = preprocessor.transform(dev_set)

# Train_data
ss_X_train= ss_train_processed_data
ss_y_train= ss_labels

# Dev_data
ss_X_dev =  ss_dev_processed_data
ss_y_dev = ss_d

# Defining a range of k values to evaluate
k_values = list(range(1, 101, 2))  # Odd numbers from 1 to 100

smarts_dev_error_rate = []
for k in k_values:  # Odd values of k
    # Create and fit the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric = 'euclidean')
    knn.fit(ss_X_train,ss_y_train)
   
    ss_y_train_pred = knn.predict(ss_X_train)
    ss_y_dev_pred = knn.predict(ss_X_dev)
    
    ss_train_error_rate = 1 - accuracy_score(ss_y_train, ss_y_train_pred)
    ss_dev_error_rate =  1 - accuracy_score(ss_y_dev, ss_y_dev_pred)
    smarts_dev_error_rate.append(ss_dev_error_rate)

    print(f"k = {k}",end='\t')
    print(f"dev_error = {ss_dev_error_rate*100:.2f}")
    


# In[11]:


import matplotlib.pyplot as plt

k_values = list(range(1, 101, 2))


plt.figure(figsize=(10, 6))

# Plotting the dev error rates for each version
plt.plot(k_values, naive_dev_error, label="Naive", marker='o')
plt.plot(k_values, smart_dev_error, label="Smart", marker='s')
plt.plot(k_values, smarts_dev_error_rate, label="Smart+Scaling", marker='x')


# Adding labels and title
plt.xlabel("k Values")
plt.ylabel("%Error")
plt.title("Dev Error Rates for Different k-NN Versions")


# Add a legend
plt.legend()

# Show the plot
plt.grid()
plt.show()


# In[ ]:




