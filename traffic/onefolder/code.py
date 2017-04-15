
# coding: utf-8

# In[1]:

import pandas as pd
dataset = pd.read_csv('/home/huseinzol05/Documents/4SEM/industrial/data.csv')

dataset.head()


# In[2]:

dataset = dataset.fillna(0)

Y = dataset.ix[:, 2:].values.mean(axis = 0)

X = range(24)


# In[4]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

plt.figure(figsize=(10,6))
plt.plot(X, Y)
plt.xlabel('Hour')
plt.ylabel('Velocity')
plt.title('Average second week')
plt.show()


# In[14]:

fig = plt.figure(figsize=(30,10))

month = ['January', 'February']
day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Average']

num = 1

for i in xrange(len(month)):
    for k in xrange(len(day)):
        
        plt.subplot(len(month), len(day), num)
        
        dataset = pd.read_csv('/home/huseinzol05/AI/visualization/traffic/onefolder/' + str(num) + '.csv')
        
        dataset = dataset.fillna(0)
        
        y = dataset.ix[:, 2:].values.mean(axis = 0)

        x = range(24)
        
        plt.plot(x, y)
        plt.xlabel('Hour')
        plt.ylabel('Velocity')
        plt.title(day[k] + ' , ' + month[i])
        
        num += 1
        
fig.tight_layout()        
plt.show() 
        


# In[ ]:



