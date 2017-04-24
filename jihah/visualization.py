
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="whitegrid", palette="muted")
current_palette = sns.color_palette()


# In[2]:

dataset = pd.read_csv("/home/huseinzol05/AI/visualization/jihah/log-CTI-fyp.csv")

# remove unimportant columns
columns = ['FileTimeUtc', 'SourceIp', 'TargetIp', 'Payload', 'SourceIpCountryCode', 'SourceIpPostalCode', 'HttpRequest',
          'HttpReferrer', 'HttpUserAgent', 'HttpMethod', 'HttpVersion', 'HttpHost', 'Custom Field 1', 'Custom Field 2',
          'Custom Field 3', 'Custom Field 4', 'Custom Field 5']

# iterate one by one to remove
for i in columns:
    del dataset[i]

# sum NaN for each rows
counter_nan = dataset.isnull().sum()
# get rows that sum of NaN == 0
counter_without_nan = counter_nan[counter_nan == 0]
# get rows that do not have NaN keys
dataset = dataset[counter_without_nan.keys()]


# Some of the left columns are string type, so lets change string into int using sklearn.labelencoder
# 
# Below I will try to normalize using t-Distributed Stochastic Neighbor Embedding (t-SNE)
# 
# ![alt text](https://raw.githubusercontent.com/huseinzol05/Introduction-Computer-Vision/master/picture/Screenshot%20from%202017-04-23%2015-16-05.png)
# 
# If you understand the equation, basically if an element has high value, probability of that element related to its population is high, and it will scattered on high value hypothesis plane.
# 
# exponent equation, totally logistic

# In[3]:

from sklearn.preprocessing import LabelEncoder

# 'spring' related data into low dimension
from sklearn.manifold import TSNE

# copy first
dataset_copy = dataset.copy()

# change strings value into int, sorted by characters
for i in xrange(dataset_copy.ix[:, :].shape[1]):
    if str(type(dataset_copy.ix[0, i])).find('str') > 0:
        dataset_copy.ix[:, i] = LabelEncoder().fit_transform(dataset_copy.ix[:, i])

labelthreat = ['High', 'Low']

# our X is first columns until second last columns
X = dataset_copy.ix[:, :-1].values

# our Y is last column
Y = dataset_copy.ix[:, -1].values

X = TSNE(n_components = 2).fit_transform(X)

fig = plt.figure(figsize = (15,15))

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(X[Y == no, 0], X[Y == no, 1], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.savefig('graph1.pdf')
plt.savefig('graph1.png') 
plt.cla()

# Wow, the data very horrible, how about we normalized and standard the data to unit variance?

# In[4]:

from sklearn.preprocessing import LabelEncoder

# 'spring' related data into low dimension
from sklearn.manifold import TSNE

# copy first
dataset_copy = dataset.copy()

# change strings value into int, sorted by characters
for i in xrange(dataset_copy.ix[:, :].shape[1]):
    if str(type(dataset_copy.ix[0, i])).find('str') > 0:
        dataset_copy.ix[:, i] = LabelEncoder().fit_transform(dataset_copy.ix[:, i])

labelthreat = ['High', 'Low']

# our X is first columns until second last columns
X = dataset_copy.ix[:, :-1].values

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

X = StandardScaler().fit_transform(X)
X = Normalizer().fit_transform(X)

# our Y is last column
Y = dataset_copy.ix[:, -1].values

X = TSNE(n_components = 2).fit_transform(X)

fig = plt.figure(figsize = (15,15))

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(X[Y == no, 0], X[Y == no, 1], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.savefig('graph2.pdf')
plt.savefig('graph2.png') 
plt.cla()

# Is it getting better? mehhhhhhhhhhh
# Let's we try to use PCA, principal component

# In[5]:

from sklearn.preprocessing import LabelEncoder

# 'spring' related data into low dimension
from sklearn.decomposition import PCA

# copy first
dataset_copy = dataset.copy()

# change strings value into int, sorted by characters
for i in xrange(dataset_copy.ix[:, :].shape[1]):
    if str(type(dataset_copy.ix[0, i])).find('str') > 0:
        dataset_copy.ix[:, i] = LabelEncoder().fit_transform(dataset_copy.ix[:, i])

labelthreat = ['High', 'Low']

# our X is first columns until second last columns
X = dataset_copy.ix[:, :-1].values

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

X = StandardScaler().fit_transform(X)
X = Normalizer().fit_transform(X)

# our Y is last column
Y = dataset_copy.ix[:, -1].values

X = PCA(n_components = 2).fit_transform(X)

fig = plt.figure(figsize = (15,15))

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(X[Y == no, 0], X[Y == no, 1], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.savefig('graph3.pdf')
plt.savefig('graph3.png') 
plt.cla()

# Okey, this one is better. Now you can understand the hypothesis plane if in 2 dimension. The data scattered nicely in PCA after normalization and standardization, so to do binary classification a lot easier.
# 
# You are correct, use logistic regression. But need to put a lot of dropout between nets to prevent overfitting and penalty system if the nets learn too fast.
# 
# Better use RELU as activation function, i think linear activation is good for this one, separating linearly during train, and do softmax logistic during predict the hypothesis.

# But what is the colleration between columns?

# In[9]:

fig = plt.figure(figsize = (30,12))

labelthreat = ['High', 'Low']

# copy first
dataset_copy = dataset.copy()

# change strings value into int, sorted by characters
for i in xrange(dataset_copy.ix[:, :].shape[1]):
    if str(type(dataset_copy.ix[0, i])).find('str') > 0:
        dataset_copy.ix[:, i] = LabelEncoder().fit_transform(dataset_copy.ix[:, i])
        
# our anchor
Y = dataset_copy.ix[:, -1].values

plt.subplot(1, 3, 1)
y = dataset_copy['Botnet'].values
x = dataset_copy['SourcePort'].values

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(x[Y == no], y[Y == no], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.ylabel('Bot net')
plt.xlabel('Source Port')
plt.title('Botnet vs Source Port')

plt.subplot(1, 3, 2)
y = dataset_copy['SourceIpAsnNr'].values

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(x[Y == no], y[Y == no], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.ylabel('SourceIpAsnNr')
plt.xlabel('Source Port')
plt.title('SourceIpAsnNr vs Source Port')


plt.subplot(1, 3, 3)
y = dataset_copy['TargetPort'].values

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(x[Y == no], y[Y == no], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.ylabel('TargetPort')
plt.xlabel('Source Port')
plt.title('TargetPort vs Source Port')

fig.tight_layout()        
plt.savefig('graph4.pdf')
plt.savefig('graph4.png') 

# i change botnet value into int
# 
# I visualized back into dictionary to make you understand.
# 
# botnet in value 1, 2, 10, 11 have HIGH THREAD CONFIDENCE

# In[13]:

dictionary = dict(zip(np.unique(dataset['Botnet'].values).tolist(), np.unique(dataset_copy['Botnet'].values).tolist()))

dictionary


# i change SourceIpAsnNr value into int
# 
# i don't know what column is this, source ip?
# 
# SourceIpAsnNr in value 0, 29, 33 have HIGH THREAD CONFIDENCE

# In[14]:

dictionary = dict(zip(np.unique(dataset['SourceIpAsnNr'].values).tolist(), np.unique(dataset_copy['SourceIpAsnNr'].values).tolist()))

dictionary


# Low TargetPort has HIGH THREAD CONFIDENCE

# In[31]:

fig = plt.figure(figsize = (30,12))

labelthreat = ['High', 'Low']

# copy first
dataset_copy = dataset.copy()

# change strings value into int, sorted by characters
for i in xrange(dataset_copy.ix[:, :].shape[1]):
    if str(type(dataset_copy.ix[0, i])).find('str') > 0:
        dataset_copy.ix[:, i] = LabelEncoder().fit_transform(dataset_copy.ix[:, i])

# our main dataset do not have 'SourceIpCity'
dataset_ipcity = pd.read_csv("/home/huseinzol05/AI/visualization/jihah/log-CTI-fyp.csv")

dataset_ipcity = dataset_ipcity['SourceIpCity']

dataset_ipcity = LabelEncoder().fit_transform(dataset_ipcity)
        
# our anchor
Y = dataset_copy.ix[:, -1].values

plt.subplot(1, 3, 1)

y = dataset_ipcity

x = dataset_copy['SourcePort'].values

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(x[Y == no], y[Y == no], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.ylabel('dataset_ipcity')
plt.xlabel('Source Port')
plt.title('SourceIpRegion vs Source Port')

plt.subplot(1, 3, 2)
y = dataset_copy['SourceIpLatitude'].values

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(x[Y == no], y[Y == no], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.ylabel('SourceIpLatitude')
plt.xlabel('Source Port')
plt.title('SourceIpLatitude vs Source Port')

plt.subplot(1, 3, 3)
y = dataset_copy['SourceIpLongitude'].values

for no, _ in enumerate(np.unique(Y)):
    plt.scatter(x[Y == no], y[Y == no], color = current_palette[no], label = labelthreat[no])
    
plt.legend()
plt.ylabel('SourceIpLongitude')
plt.xlabel('Source Port')
plt.title('SourceIpLongitude vs Source Port')

fig.tight_layout()        
plt.savefig('graph5.pdf')
plt.savefig('graph5.png') 

# In[34]:

dataset_ipcity = pd.read_csv("/home/huseinzol05/AI/visualization/jihah/log-CTI-fyp.csv")

dataset_ipcity = dataset_ipcity['SourceIpCity'].values

label_city = np.unique(dataset_ipcity).tolist()

dataset_ipcity = LabelEncoder().fit_transform(dataset_ipcity)

city = np.unique(dataset_ipcity).tolist()

dictionary = dict(zip(label_city, city))

dictionary


# You judge by yourself. haha
