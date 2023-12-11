#!/usr/bin/env python
# coding: utf-8

# In[33]:


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


# In[34]:


blind_data = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.test.blind",header = None, 
                         names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                                'hours-per-week', 'country-of-origin', 'target'])


# In[35]:


train_data = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.train.txt.5k",header = None, 
                         names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 
                                'hours-per-week', 'country-of-origin', 'target'])


# In[36]:


# Train_Data
labels = train_data['target']
train_features = train_data.drop(['target'], axis = 1)


# In[37]:


blind_features = blind_data.drop(['target'], axis = 1)


# In[38]:


#blind_data.shape


# In[39]:


#blind_features.shape


# In[40]:


#num_processor = 'passthrough' # i.e., no transformation
num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[41]:


preprocessor = ColumnTransformer([
('num', num_processor, ['age','hours-per-week']),
('cat', cat_processor, ['sector','education', 'marital-status', 'occupation', 'race', 'sex','country-of-origin'])
])
preprocessor.fit(train_data)
train_processed_data = preprocessor.transform(train_features)
blind_processed_data = preprocessor.transform(blind_features)


# In[42]:


# Train Data
X_train= train_processed_data
y_train= labels


# Blind_data
X_test_blind =  blind_processed_data


# In[43]:


# SELF MNAHATTAN K-NN ALGORITHM



def custom_manhattan_distance(a, b):
#Calculating the Manhattan distance between two data points 'a' and 'b'
    distance = (np.sum(np.abs(a - b),axis=1))
    return distance



k = 41
y_test_pred=[]
for query_point in X_test_blind:
    distances = custom_manhattan_distance(X_train,query_point)
    nearest_indices = np.argsort(distances)[:k]
    nearest_neighbors = y_train[nearest_indices]
    most_common_label = Counter(nearest_neighbors).most_common(1)[0][0]
    y_test_pred.append(most_common_label)

print(f"k = {k}",end='\t')
print("positive_ratio = ",(y_test_pred.count(" >50K")/ len(y_test_pred))* 100)
blind_data['target'] = y_test_pred
#labels['target'] = y_test_pred


# In[44]:


#blind_data


# In[45]:


output_file = "C:/3rd term/ML/HW-1/hw1-data/income.test.predicted"


# In[46]:


with open(output_file, 'w') as file:
    for i in range(len(blind_data)):
        # Extract the feature values
        features = blind_data.iloc[i].values.tolist()
       
        feature_str = ', '.join(map(str, features)).replace(" ","")
        feature_str_list = feature_str.split(",")
        formatted_features = ", ".join(feature_str_list) + '\n'
       
        file.write(formatted_features)
   
        print(formatted_features, end='\n')

print(f'Predictions saved to {output_file}')
       


# In[ ]:




