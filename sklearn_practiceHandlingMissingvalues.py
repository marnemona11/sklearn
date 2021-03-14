#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
datapath = r"E:\Monali\Class\Datasets\archive\reddit_wsb.csv"
df = pd.read_csv(datapath)
df


# In[7]:


df.shape


# # Handling numerical missing values

# In[8]:


num_vars =df.select_dtypes(include=["int64","float64"]).columns
num_vars


# In[12]:


#check null values
df.isnull()


# In[13]:


df.isnull().sum()


# # checking null values in all columns

# In[14]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

df.isnull().sum()


# In[15]:


#Check total null values
df.isnull().sum().sum()


# # dropping  null values containing column

# In[16]:


df_New = df.drop(columns="body")
df_New.shape


# In[18]:


imputer_mean = SimpleImputer(strategy='mean')
imputer_mean.fit(df[num_vars])


# In[19]:


imputer_mean.statistics_


# In[20]:


imputer_mean.transform(df[num_vars])


# In[22]:


df_New[num_vars].isnull()


# In[23]:


df_New[num_vars].isnull().sum()


# In[24]:


df_New[num_vars].isnull().sum().sum()


# # Handle String missing data

# In[26]:


cat_vars = df.select_dtypes(include=["O"]).columns
cat_vars


# In[27]:


df[cat_vars].isnull().sum()


# In[28]:


df[cat_vars].isnull().sum().sum()


# In[29]:


imputer_mode = SimpleImputer(strategy='most_frequent')
imputer_mode


# In[30]:


imputer_mode.fit(df[cat_vars])


# In[32]:


imputer_mode.statistics_


# In[33]:


df[cat_vars].isnull().sum().sum()


# In[ ]:




