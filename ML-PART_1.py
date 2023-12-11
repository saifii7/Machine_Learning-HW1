#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


toy = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/toy.txt")


# In[3]:


toy


# In[4]:


names = ["age","sector"]


# In[5]:


toy = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/toy.txt", names = names )


# In[6]:


toy


# In[7]:


toy['age']


# In[8]:


toy['sector']


# In[9]:


x = pd.get_dummies(toy['age'])


# In[10]:


y = pd.get_dummies(toy['sector'])


# In[11]:


age = np.array(x)


# In[12]:


sector = np.array(y)


# In[13]:


np.concatenate((age,sector), axis = 1)


# In[14]:


train_data = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.train.txt.5k",header = None, names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'country-of-origin', 'target'])


# In[15]:


dev_set = pd.read_csv("C:/3rd term/ML/HW-1/hw1-data/income.dev.txt",header = None, names=['age', 'sector', 'education', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'country-of-origin', 'target'])


# In[16]:


dev_set


# In[17]:


train_data


# In[18]:


train_positive_percentage = (train_data['target'] == ' <=50K').mean() * 100


# In[19]:


dev_positive_percentage = (dev_set['target'] == ' <=50K').mean() * 100


# QUESTION-1 :- What percentage of the training data has a positive label (>50K)? (This is known as the positive %). What
# about the dev set? Does it make sense given your knowledge of the average US per capita income?

# In[20]:


print(f"Positive Percentage in Training Data: {train_positive_percentage:.2f}%")
print(f"Positive Percentage in Dev Data: {dev_positive_percentage:.2f}%")


# QUESTION-2 :- What are the youngest and oldest ages in the training set? What are the least and most amounts of hours
# per week do people in this set work?

# In[45]:



youngest_age = train_data["age"].min()
oldest_age = train_data["age"].max()


# In[23]:


print(youngest_age)


# In[24]:


print(oldest_age)


# In[25]:


least_hr = train_data["hours-per-week"].min()
most_hr = train_data["hours-per-week"].max()


# In[26]:


print(least_hr)


# In[27]:


print(most_hr)


# QUESTION-5 :- How many features do you have in total (i.e., the dimensionality)?

# In[30]:


#row 2 = 7
#row 3 = 16
#row 4 = 7
#row 5 = 14
#row 6 = 5
#row 7 = 2
#row 9 = 39
#and now adding two numerical field which result in = 91

