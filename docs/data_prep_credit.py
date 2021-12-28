# Credit data preparation

# The credit data is a simulated data set containing information on ten thousand customers (taken from {cite:t}`James2021`). The aim here is to use a classification model to predict which customers will default on their credit card debt (i.e., failure to repay a debt):
# 
# - default: A categorical variable with levels No and Yes indicating whether the customer defaulted on their debt 
# - student: A categorical variable with levels No and Yes indicating whether the customer is a student
# - balance: The average balance that the customer has remaining on their credit card after making their monthly payment
# - income: Income of customer

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/kirenz/classification/main/_static/data/Default.csv')

dummies = pd.get_dummies(df[['default', 'student']], drop_first=True, dtype=float)
dummies.head(3)

# combine data and drop original categorical variables
df = pd.concat([df, dummies], axis=1).drop(columns = ['default', 'student'])
df.head(3)

# Next, we create our y label and features:
y = df.default_Yes
X = df.drop(columns = 'default_Yes')

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)