#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# 
# This code tutorial is mainly based on Python code provided by [Jordi Warmenhoven](https://github.com/JWarmenhoven/ISLR-python). To learn more about the method, take a look at ["An Introduction to Statistical Learning"](https://www.statlearning.com/) by James et al. (2021). 

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


# ## Logistic regression

# ### Model

# In[4]:


import sklearn.linear_model as skl_lm

clf = skl_lm.LogisticRegression()
y_pred = clf.fit(X_train, y_train).predict(X_test)


# In[5]:


# Return the mean accuracy on the given test data and labels:
clf.score(X_test, y_test)


# ### Confusion matrix

# In[6]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# ### Classification report

# In[7]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))


# ### Change threshold

# Use specific threshold

# In[8]:


pred_proba = clf.predict_proba(X_test)

df_ = pd.DataFrame({'y_test': y_test, 'y_pred': pred_proba[:,1] > .25})
cm = confusion_matrix(y_test, df_['y_pred'])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# ### Classification report

# In[9]:



print(classification_report(y_test, df_['y_pred']))

