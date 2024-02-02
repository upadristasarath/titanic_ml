#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

def get_title(name):
    if '.' in name:
        return name.split(",")[1].split(".")[0].strip()
    else:
        return "No titles in name"

def shorter_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer', 'Don',  "the Countess", "Done", "Lady", "Sir"]:
        return "Royality"
    elif title == "Mme":
        return "Mrs"
    elif title in ["Mlle", "Ms"]:
        return "Miss"
    else:
        return title

df = pd.read_csv("train.csv")
df['Title'] = df['Name'].map(lambda x: get_title(x))
df['Title'] = df.apply(shorter_titles, axis=1)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)

df.drop("Cabin", axis=1, inplace=True)
df.drop("Ticket", axis=1, inplace=True)
df.drop("Name", axis=1, inplace=True)
df.Sex.replace(('male', 'female'), (0,1), inplace=True)
df.Embarked.replace(('S', 'C', 'Q'), (0, 1, 2), inplace=True)
df.Title.replace(("Mr", "Miss", "Mrs", "Master", "Dr", "Rev", "Royality", "Officer", "Dona"), (0, 1, 2, 3, 4, 5, 6, 7, 8), inplace=True)

x = df.drop(['Survived', 'PassengerId'], axis = 1)
y = df["Survived"]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1)
randomforest = RandomForestClassifier()
x_train = x_train.values
randomforest.fit(x_train, y_train)

pickle.dump(randomforest, open('titanic_model.sav', 'wb'))


# In[14]:


# Correct order in the dataframe
def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
  import pickle
  x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
  randomforest = pickle.load(open('titanic_model.sav', 'rb'))
  predications = randomforest.predict(x)
  print(predications)


# In[16]:


prediction_model(1, 1, 19, 2, 1, 2, 1, 2)


# In[ ]:




