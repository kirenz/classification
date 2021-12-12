#!/usr/bin/env python
# coding: utf-8

# # Discriminant Analysis
# 
# This code tutorial is mainly based on Python code provided by [Jordi Warmenhoven](https://github.com/JWarmenhoven/ISLR-python). To learn more about discriminant analysis, take a look at ["An Introduction to Statistical Learning"](https://www.statlearning.com/) by James et al. (2021). 
# 
# If you are already familiar with Principal Component Analysis (PCA), note that Discriminant Analysis is similar to PCA:
# 
# - both reduce the dimensions in our data
# - PCA identifies variables with the most variation
# - Discriminant Analysis maximizes the separation of some categorical labels

# ## Data
# 
# A data frame with 10000 observations on the following 4 variables:
# 
# - default: A categorical variable with levels No and Yes indicating whether the customer defaulted on their debt
# 
# - student: A categorical variable with levels No and Yes indicating whether the customer is a student
# 
# - balance: The average balance that the customer has remaining on their credit card after making their monthly payment
# 
# - income: Income of customer

# In[1]:


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/kirenz/classification/main/_static/data/Default.csv')

# Note: factorize() returns two objects: a label array and an array with the unique values.
# We are only interested in the first object. 
df['default2'] = df.default.factorize()[0]
df['student2'] = df.student.factorize()[0]
df.head(3)


# In[2]:


X = df[['balance', 'income', 'student2']]
y = df.default2


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# ## Linear discriminant analysis

# ### Model

# In[4]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(solver='svd')
y_pred = clf.fit(X_train, y_train).predict(X_test)


# In[5]:


clf.coef_


# ### Confusion matrix

# In[6]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# Confusion matrix as pandas table:

# In[7]:


df_ = pd.DataFrame({'True default status': y_test,
                    'Predicted default status': y_pred})
                    
df_.replace(to_replace={0:'No', 1:'Yes'}, inplace=True)
df_.groupby(['Predicted default status','True default status']).size().unstack('True default status')


# ### Classification report

# In[8]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))


# ### Change threshold

# Use specific threshold

# In[9]:


# Obtain probabilities 
y_prob = clf.fit(X_train, y_train).predict_proba(X_test)


# In[10]:


# Set threshold 
decision_prob = 0.2

# Build confusion matrix
df_ = pd.DataFrame({'True default status': y_test,
                    'Predicted default status': y_prob[:,1] > decision_prob})

df_.replace(to_replace={0:'No', 1:'Yes', 'True':'Yes', 'False':'No'}, inplace=True)
df_.groupby(['Predicted default status','True default status']).size().unstack('True default status')


# In[11]:



print(classification_report(df_['True default status'], df_['Predicted default status'], target_names=['No', 'Yes']))


# ## Quadratic Discriminant Analysis

# ### Model

# In[12]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis()
y_pred = clf.fit(X_train, y_train).predict(X_test)


# ### Confusion matrix

# In[13]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# ### Classification report

# In[14]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, digits=3))

