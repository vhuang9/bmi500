#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[46]:


# 1) returns first n terms of Leibniz formula
def leibniz1(n):
    leibSum = 0
    for k in range(n):
        leibSum += (-1)**k/(2*k+1)
    return leibSum*4


# In[47]:


# 2a) use a for-loop and if-statement with modulo operator to determine whether to add or subtract terms
def leibniz2a(n):
    leibSum = 0
    for k in range(n):
        if k%2 > 0:
            leibSum -= 1/(2*(k+1)-1)
        else:
            leibSum += 1/(2*(k+1)-1)
    return leibSum*4


# In[48]:


# 2b) use a for-loop with quantity (-1)**n to determine whether to add or subtract terms
def leibniz2b(n):
    leibSum = 0
    for k in range(n):
        if (-1)**k < 0:
            leibSum -= 1/(2*(k+1)-1)
        else:
            leibSum += 1/(2*(k+1)-1)
    return leibSum*4


# In[49]:


# 2c) construct a list and compute the sum of the terms in the list
def leibniz2c(n): 
    leibList = [(-1)**k/(2*k+1) for k in range(0,n)]
    print(leibList)
    return sum(leibList)*4


# In[40]:


# 2d) construct a set and compute the sum of the terms in the set
def leibniz2d(n): 
    leibSet = set([(-1)**k/(2*k+1) for k in range(0,n)])
    print(leibSet)
    return sum(leibSet)*4


# In[41]:


# 2e) construct a dictionary and compute the sum of the terms in the dictionary
def leibniz2e(n): 
    leibDict = {k: term for k,term in 
                enumerate([(-1)**key/(2*key+1) for key in range(0,6)])}
    print(leibDict)
    return sum(leibDict.values())*4


# In[42]:


# 2f) construct a numpy array and compute the sum of the terms in the array
def leibniz2f(n): 
    leibArray = np.arange(0,n) 
    leibArray = (-1)**leibArray/(2*leibArray+1)
    print(leibArray)
    return sum(leibArray)*4


# In[43]:


# 2g) construct a numpy array and separately compute sum of positive and negative terms then add them together
#     use array indexing
def leibniz2g(n): 
    leibArray = np.arange(0,n) 
    leibArray = (-1)**leibArray/(2*leibArray+1)
    posSum = sum(leibArray[::2])
    negSum = sum(leibArray[1::2])
    print(leibArray)
    return (posSum+negSum)*4


# In[44]:


# 2j) combine first and second, third and fourth terms, etc to change series from an alternating to a
#     non-alternating series and compute sum of combined terms
def leibniz2j(n):
    leibArray = np.arange(0,n,2)
    leibArray = 1/(2*leibArray+1) - 1/(2*(leibArray+1)+1)
    print(leibArray)
    return sum(leibArray)*4


# In[51]:


x = 8
print(leibniz1(x))
print(leibniz2a(x))
print(leibniz2b(x))
print(leibniz2c(x))
print(leibniz2d(x))
print(leibniz2e(x))
print(leibniz2f(x))
print(leibniz2g(x))
print(leibniz2j(x))
print((1-1/3+1/5-1/7+1/9-1/11)*4)


# In[73]:


(np.pi/4-leibniz1(6))/(np.pi/4)


# In[81]:


# Plot absolute error in sum as a function of the number of terms in the sum

n = 6
true = np.pi/4
# error = (pi/4-leib)/(pi/4)
error = np.array([abs((true-leibniz1(k))/true) for k in range(0,n)])
print(error)
plt.figure(figsize=(10,10))
plt.title('# terms vs. error')
plt.plot(np.arange(1,n+1),error,'-o')


# In[69]:


np.arange(1,10)


# In[ ]:





# In[28]:


# dictionary notes

# import string
# dict1 = {value: (int(key) + 1) for key, value in 
# enumerate(list(string.ascii_lowercase))}


{(int(key) + 1):value for key, value in 
enumerate([(-1)**key/(2*key+1) for key in range(0,6)])}

{(key,value) for (key,value) in zip(key_list,value_list)}


# In[59]:


(-1)**(range(1,4+1))


# In[10]:


m = 6
sum([(-1)**k/(2*k+1) for k in range(0,m)])


# In[87]:


l = np.arange(0,6)
l


# In[91]:


type(sum(l.tolist()))


# In[86]:


(-1)**l/((2*l)+1)


# In[67]:


(-1)**1/(2*-1+1)

