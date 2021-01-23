#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


n_trials = 100
n_neighbours = [2, 3, 4, 5, 6]



df = pd.read_csv('../heart_failure_clinical_records_dataset.csv')
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']


#df.head()



def draw_rand_nums(n_rand):
    """
    Returns a list of n_rand random numbers
    drawn between 0 and 100000. Useful to 
    create different train and test splits. 
    """
    rand_nums = []
    for i in range(n_rand):
        r = np.random.randint(1, 100000)
        rand_nums.append(r)
    return rand_nums


def make_pipes(k):
    """
    Creates four pipelines and returns them 
    as in a list. 
    """
    pipe0 = Pipeline([('KNN           ', KNeighborsClassifier(n_neighbors=k))])
    pipe1 = Pipeline([('MinMaxScaler  ', MinMaxScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    pipe2 = Pipeline([('RobustScaler  ', RobustScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    pipe3 = Pipeline([('StandardScaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])

    pipes = [pipe0, pipe1, pipe2, pipe3]
    return pipes

def get_scores(pipes, display=False):
    """"
    Finds score for each pipeline in 'pipes' list.
    Returns a list of scores. If display=True, 
    each score is also printed to the screen.
    """
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



"""
1. n_trials random numbers are created. 
2. For each n_neighbours value (number of neighbors=k)
   train-test split is performed.
3. Each model (kNN alone, kNN with different scalings),
   model is trained and tested using Pipelines. 
4. Scores are stored in a list. 

The random numbers are used as random seed while 
splitting data into train and test data sets. 
Please note, cross validation is not done even though 
it is better and more efficient. 

In this version, train-test split is done inside 
the "k in n_neighbors" loop. This is not optimal.
But for a small data set, this works fine. 
"""

rand_nums = draw_rand_nums(n_trials)
scores_df = pd.DataFrame()

for k in n_neighbours:
    pipes = make_pipes(k)
    
    scores = []
    for r in rand_nums:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r)
        scores.append(get_scores(pipes))

    scores_df["k="+str(k)] = pd.DataFrame(scores).mean()

scores_df.index = ["pipe=" + str(i) for i in scores_df.index ]
scores_df.style.set_caption("Average Score")






