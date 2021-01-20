#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


# In[2]:


n_trials = 100
n_neighbours = [2, 3, 4, 5, 6]


# In[3]:


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']


# In[4]:


df.head()


# In[5]:


def draw_rand_nums(n_rand):
    rand_nums = []
    for i in range(n_rand):
        r = np.random.randint(1, 100000)
        rand_nums.append(r)
    return rand_nums


def make_pipes(k):
    #n_neighbors = 3
    pipe0 = Pipeline([('KNN           ', KNeighborsClassifier(n_neighbors=k))])
    pipe1 = Pipeline([('MinMaxScaler  ', MinMaxScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    pipe2 = Pipeline([('RobustScaler  ', RobustScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    pipe3 = Pipeline([('StandardScaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])

    pipes = [pipe0, pipe1, pipe2, pipe3]
    return pipes

def get_scores(pipes, display=False):
    '''
    Finds score for each pipeline in 'pipes' list.
    Returns a list of scores.
    If display=True, each score is also printed.
    '''
    scores = []
    for p in pipes:
        p.fit(X_train, y_train)
        score = p.score(X_test, y_test)
        score = np.round(score, 2)
        scores.append(score)
        name = p.steps[0][0]
        if display:
            print(name, score)
    return scores


# In[9]:


rand_nums = draw_rand_nums(n_trials)
scores_df = pd.DataFrame()

for k in n_neighbours:
    pipes = make_pipes(k)
    
    scores = []
    for r in rand_nums:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r)
        scores.append(get_scores(pipes))

    scores_df["k="+str(k)] = pd.DataFrame(scores).mean()
#    print("k = ", k)
#    print(pd.DataFrame(scores).mean())


# In[10]:


scores_df


# In[8]:


scores_df.T.max()


# In[ ]:





# In[ ]:




