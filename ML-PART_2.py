#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
from collections import Counter
import math
import warnings
warnings.filterwarnings('ignore')


# In[55]:


train_data = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.train.txt.5k",header = None, 
                         names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                                'hours-per-week', 'country-of-origin', 'target'])


# In[56]:


dev_set = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.dev.txt",header = None, 
                      names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                             'hours-per-week', 'country-of-origin', 'target'])


# QUESTION-3 :- After implementing the naive binarization to the real training set (NOT the toy set), what is the
# feature dimension? Does it match with the result from Part 1 Q5?

# In[57]:


encoder.fit(train_data) 
binary_data = encoder.transform(train_data)


# In[58]:


encoder.get_feature_names_out()


# In[59]:


# Train_Data
labels = train_data['target']
b = train_data.drop(['target'],axis = 1)


# In[60]:


# Dev_set
d = dev_set['target']
d_features = dev_set.drop(['target'], axis = 1)


# In[61]:


# Train Binary Conversion
encoder.fit(b)
feature = encoder.transform(b)

# Dev Binary conversion
d_f = encoder.transform(d_features)


# In[62]:


X_train= feature
y_train= labels


# In[63]:


X_dev = d_f
y_dev = d


# In[64]:


#X_dev.shape


# QUESTION-4 :- Fit k-NN via Scikit-Learn

# In[78]:


k_values = list(range(1, 101, 2)) 

for k in k_values:  
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
   

    y_train_pred = knn.predict(X_train)
    y_dev_pred = knn.predict(X_dev)
    
    train_error_rate = 1 - accuracy_score(y_train, y_train_pred)
    dev_error_rate =  1 - accuracy_score(y_dev, y_dev_pred)
    
    y_train_pred = list(y_train_pred)
    y_dev_pred = list(y_dev_pred)
    
    train_count_50k = y_train_pred.count(" >50K")
    train_positive_ratio = (train_count_50k / len(y_train_pred))
    
    dev_count_50k = y_dev_pred.count(" >50K")
    dev_positive_ratio = (dev_count_50k / len(y_dev_pred))
    
    
    print(f"k = {k}",end='\t')
    print(f"train_err = {train_error_rate*100:.2f}",end='\t')
    print(f"Train Positive Ratio = {train_positive_ratio*100:.2f}", end ='\t')
    print(f"dev_err = {dev_error_rate*100:.2f}",end='\t\t')
    print(f"Dev Positive Ratio = {dev_positive_ratio*100:.2f}")
   


# In[ ]:




