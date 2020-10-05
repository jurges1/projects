#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


height_weight_data = pd.read_csv('datasets/500_Person_Gender_Height_Weight_Index.csv')

height_weight_data.head()


# In[3]:


height_weight_data.drop('Index', inplace=True, axis=1)


# In[4]:


height_weight_data.shape


# In[5]:


num_records = height_weight_data.shape[0]

num_records


# In[6]:


height_data = height_weight_data[['Height']].copy()

height_data.head()


# In[7]:


weight_data = height_weight_data[['Weight']].copy()

weight_data.head()


# In[8]:


counts = [1] * num_records

height_data['counts_height'] = counts
weight_data['counts_weight'] = counts


# In[9]:


weight_data = weight_data.sort_values('Weight')

weight_data.tail()


# In[10]:


height_data = height_data.sort_values('Height')

height_data.tail()


# In[11]:


height_data = height_data.groupby('Height', as_index=False).count()

height_data.head(10)


# In[12]:


weight_data = weight_data.groupby('Weight', as_index=False).count()

weight_data.head(10)


# In[13]:


height_data['cumcounts_height'] = height_data['counts_height'].cumsum()

height_data.head(10)


# In[14]:


weight_data['cumcounts_weight'] = weight_data['counts_weight'].cumsum()

weight_data.head(10)


# ### Interquartile Range

# In[15]:


q1_height = height_weight_data['Height'].quantile(.25)

q1_height


# In[16]:


q3_height = height_weight_data['Height'].quantile(.75)

q3_height


# In[17]:


iqr_height = q3_height - q1_height

iqr_height


# In[18]:


plt.figure(figsize=(12, 8))

height_weight_data['Height'].hist(bins=30)

plt.axvline(q1_height, color='r', label='Q1')
plt.axvline(q3_height, color='g', label='Q2')

plt.legend()


# In[19]:


plt.figure(figsize=(12, 8))

height_weight_data['Weight'].hist(bins=30)

plt.axvline(height_weight_data['Weight'].quantile(.25), color='r', label='Q1')
plt.axvline(height_weight_data['Weight'].quantile(.75), color='g', label='Q2')

plt.legend()


# In[20]:


plt.figure(figsize=(12, 8))

plt.scatter(height_weight_data['Weight'], height_weight_data['Height'], s=100)

plt.axvline(height_weight_data['Weight'].quantile(.25), color='r', label='Q1 Weight')
plt.axvline(height_weight_data['Weight'].quantile(.75), color='g', label='Q2 Weight')

plt.axhline(height_weight_data['Height'].quantile(.25), color='y', label='Q1 Height')
plt.axhline(height_weight_data['Height'].quantile(.75), color='m', label='Q2 Height')

plt.legend()


# In[21]:


plt.figure(figsize=(12, 8))

plt.bar(height_data['Height'], height_data['cumcounts_height'])

plt.axvline(height_weight_data['Height'].quantile(.25), color='y', label='25%')
plt.axvline(height_weight_data['Height'].quantile(.50), color='m', label='50%')
plt.axvline(height_weight_data['Height'].quantile(.75), color='r', label='75%')


# In[22]:


plt.figure(figsize=(12, 8))

plt.bar(height_data['Height'], height_data['cumcounts_height'])

plt.axvline(height_weight_data['Height'].quantile(.25), color='y', label='25%')
plt.axvline(height_weight_data['Height'].quantile(.50), color='m', label='50%')
plt.axvline(height_weight_data['Height'].quantile(.75), color='r', label='75%')

plt.axhline(.25 * num_records, color='y', label='25%')
plt.axhline(.5 * num_records, color='m', label='50%')
plt.axhline(.75 * num_records, color='r', label='75%')


# ### Calculating Variance

# In[23]:


def variance(data):
    
    diffs = 0
    avg = sum(data) / len(data)
    
    for n in data:
        diffs += (n - avg)**2
    
    return (diffs/(len(data)-1))


# In[24]:


variance(height_weight_data['Height'])


# In[25]:


variance(height_weight_data['Weight'])


# In[26]:


height_weight_data['Height'].var()


# In[27]:


height_weight_data['Weight'].var()


# ### Standard Deviation

# In[28]:


std_height = (variance(height_weight_data['Height'])) ** 0.5

std_height


# In[29]:


std_weight = (variance(height_weight_data['Weight'])) ** 0.5

std_weight


# In[30]:


height_weight_data['Height'].std()


# In[31]:


height_weight_data['Weight'].std()


# In[32]:


weight_mean = height_weight_data['Weight'].mean()

weight_std = height_weight_data['Weight'].std()


# In[33]:


plt.figure(figsize=(12, 8))

height_weight_data['Weight'].hist(bins=20)

plt.axvline(weight_mean, color='r', label='mean')

plt.axvline(weight_mean - weight_std, color='g', label='1 standard deviation')
plt.axvline(weight_mean + weight_std, color='g', label='1 standard deviation')

plt.legend()


# In[34]:


listOfSeries = [pd.Series(['Male', 40, 30], index=height_weight_data.columns ), 
                pd.Series(['Female', 66, 37], index=height_weight_data.columns ), 
                pd.Series(['Female', 199, 410], index=height_weight_data.columns ),
                pd.Series(['Male', 202, 390], index=height_weight_data.columns ), 
                pd.Series(['Female', 77, 210], index=height_weight_data.columns ),
                pd.Series(['Male', 88, 203], index=height_weight_data.columns )]


# In[35]:


height_weight_updated = height_weight_data.append(listOfSeries , ignore_index=True)

height_weight_updated.tail()


# In[36]:


plt.figure(figsize=(12, 8))

height_weight_updated['Weight'].hist(bins=100)


# In[37]:


plt.figure(figsize=(12, 8))

height_weight_updated['Height'].hist(bins=100)


# In[38]:


height_weight_updated['Height'].quantile(.25)


# In[39]:


q1_height


# In[40]:


height_weight_updated['Height'].quantile(.75)


# In[41]:


q3_height

