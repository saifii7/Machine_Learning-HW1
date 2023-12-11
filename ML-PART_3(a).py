#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[11]:


train_data = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.train.txt.5k",header = None, 
                         names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                                'hours-per-week', 'country-of-origin', 'target'])


# In[12]:


dev_set = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.dev.txt",header = None, 
                      names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                             'hours-per-week', 'country-of-origin', 'target'])


# In[13]:


# Train_Data
labels = train_data['target']


# Dev_set
d = dev_set['target']


# In[14]:


num_processor = 'passthrough' # i.e., no transformation
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[15]:


preprocessor = ColumnTransformer([
('num', num_processor, ['age','hours-per-week']),
('cat', cat_processor, ['sector','education', 'marital-status', 'occupation', 'race', 'sex','country-of-origin'])
])
preprocessor.fit(train_data)
train_processed_data = preprocessor.transform(train_data)
dev_processed_data = preprocessor.transform(dev_set)


# In[16]:


# Train_data
X_train= train_processed_data
y_train= labels

# Dev_data
X_dev =  dev_processed_data
y_dev = d


# In[19]:


k_values = list(range(1, 101, 2))  


error_rates = []
predicted_positive_rates = []


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
    
    #print(f"k = {k}",end='\t')
    #print(f"train_error = {train_error_rate*100:.2f}",end='\t')
    #print(f"training_accuracy+ % = {train_accuracy*100:.2f}",end='\t')

    #print(f"dev_error = {dev_error_rate*100:.2f}",end='\t')
    #print(f"dev + % = {dev_accuracy*100:.2f}")


# In[ ]:




