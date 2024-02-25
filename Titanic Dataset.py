#!/usr/bin/env python
# coding: utf-8

# # Titanic dataset analysys by KODI VENU

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
data = pd.read_csv('titanic.csv')


# Displaying Top 10 Rows of the Dataset

# In[2]:


data.head(10)


# Displaying Last 10 Rows of the Dataset

# In[3]:


data.tail(10)


# Finding shape of the Dataset

# In[4]:


data.shape


# In[5]:


print("Number of rows", data.shape[0])
print("Number of columns", data.shape[1])


# Dataset Overall Statistics

# In[6]:


data.describe()


# In[7]:


data.describe(include='all')


# In[8]:


data.info()


# Data filtering

# In[9]:


data.columns


# In[10]:


data['Name']


# In[11]:


data[['Name','Age']]


# In[12]:


data['Sex']=='male'


# In[13]:


sum(data['Sex']=='male')


# In[14]:


data[data['Sex']=='male']


# In[15]:


data[data['Sex']=='male'].head()


# In[16]:


data.columns


# In[17]:


data['Survived']==1


# In[18]:


sum(data['Survived']==1)


# In[19]:


data[data['Survived']==1]


# Checking Null Values in the dataset

# In[20]:


data.isnull()


# In[21]:


data.isnull().sum()


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[23]:


sns.heatmap(data.isnull())


# In[24]:


per_missing=data.isnull().sum()*100/len(data)
per_missing


# drop the column

# In[25]:


data=data.drop('Cabin',axis=1)


# In[26]:


data.isnull().sum()


# Handling missing values

# In[27]:


data.columns


# In[28]:


data['Embarked']


# In[29]:


data['Embarked'].mode()


# In[30]:


data['Embarked'].fillna('s',inplace=True)


# In[31]:


data.isnull().sum()


# In[32]:


data['Age']


# In[33]:


data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[34]:


data.isnull().sum()


# Categorical data encoding

# In[35]:


data.head()


# In[36]:


data['Sex'].unique()


# In[37]:


data['Gender']=data['Sex'].map({'male':1,'female':0})


# In[38]:


data.head(1)


# In[39]:


x=data['Sex'].map({'male':1,'female':0})


# In[40]:


data.insert(5,'Gender_New',x)


# In[41]:


data.head(1)


# In[42]:


data['Embarked'].unique()


# In[43]:


data['Embarked'].nunique()


# In[44]:


data['Embarked']=data['Embarked'].replace('s','S')


# In[45]:


data['Embarked'].unique()


# In[46]:


data['Embarked'].nunique()


# In[47]:


pd.get_dummies(data,columns=['Embarked'])


# In[48]:


data1=pd.get_dummies(data,columns=['Embarked'],drop_first=True)


# In[49]:


data1.head(1)


# What is univariate analysis?
# How many people survived and how many died?
# How many passengers were in first class, second class, third class?
# Number of male and female passengers?

# In[50]:


data.columns


# In[51]:


data['Survived'].value_counts()


# In[52]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[54]:


sns.countplot(data['Survived'])


# In[55]:


data.columns


# In[56]:


data['Pclass'].value_counts()


# In[57]:


sns.countplot(data['Pclass'])


# In[58]:


data.columns


# In[59]:


data['Sex'].value_counts()


# In[67]:


sns.countplot(data['Sex'])


# In[65]:


plt.hist(data['Age'])


# In[68]:


sns.boxplot(data['Age'])


# In[71]:


sns.boxplot(data['Age'],orient='h')


# Bivariate Analysis?
# How has better chance of survival male or female?
# Which passenger class has better chance of survival(first,second or third class)?

# In[72]:


data.columns


# In[73]:


sns.barplot(x='Sex',y='Survived',data=data)


# In[74]:


sns.barplot(x='Pclass',y='Survived',data=data)


# Feature engineering

# In[75]:


data.columns


# In[76]:


data['Family_Size'] = data['SibSp']+data['Parch']


# In[77]:


data.head(1)


# In[78]:


data.columns


# In[79]:


data['Fare_Per_Person']=data['Fare']/(data['Family_Size']+1)


# In[80]:


data.head(1)


# In[ ]:




