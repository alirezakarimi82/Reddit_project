#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import recall_score


# In[ ]:


df = pd.read_csv('./datasets/data_all.csv')


# In[ ]:


X = df['content']
y = df['subreddit']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)


# In[ ]:


vects = [CountVectorizer(), TfidfVectorizer()]
models = [LogisticRegression(), KNeighborsClassifier(),
          MultinomialNB(), RandomForestClassifier(random_state=42),
          AdaBoostClassifier(random_state=42),
          GradientBoostingClassifier(random_state=42),
          SVC(random_state=42)]


# In[ ]:


pipes = []
for vect in vects:
    for model in models:
        pipe = Pipeline([
        ('vect', vect),
        ('model', model)])
        pipes.append(pipe)


# In[ ]:


params_vect = {'CountVectorizer': {'vect__stop_words': [None, 'english'],
                                   'vect__ngram_range': [(1,1), (1,2)]},
               'TfidfVectorizer': {'vect__stop_words': [None, 'english'],
                                   'vect__ngram_range': [(1,1), (1,2)]}}


# In[ ]:


params_model = {'LogisticRegression': {'model__penalty': ['l1', 'l2'],
                                       'model__C': np.logspace(-3, 3, 10)},
                'KNeighborsClassifier': {'model__n_neighbors': [5, 7, 15],
                                         'model__weights': ['uniform', 'distance']},
                'MultinomialNB': {},
                'RandomForestClassifier': {'model__n_estimators': [100, 150, 200],
                                           'model__max_depth': [None, 2, 4]},
                'AdaBoostClassifier': {'model__n_estimators': [100, 150, 200]},
                'GradientBoostingClassifier': {'model__n_estimators': [100, 150, 200]},
                'SVC': {'model__C': np.logspace(-2, 2, 5),
                        'model__gamma': np.logspace(-4, 0, 5)}
               }


# In[ ]:


df = pd.DataFrame(columns=['Vectorizer', 'Classifier',
                           'best_params', 'train_recall', 'test_recall',
                           'train_accuracy', 'test_accuracy'])


# In[ ]:


for pipe in pipes:
    Vectorizer = str(pipe.steps[0][1]).split('(')[0]
    Classifier = str(pipe.steps[1][1]).split('(')[0]

    print('\nGrid search using {} and {} ...\n'.format(Vectorizer, Classifier))

    params = {**params_vect[Vectorizer], **params_model[Classifier]}

    grid = GridSearchCV(pipe,
                    param_grid=params,
                    cv = 5,
                    scoring = 'accuracy',
                    verbose = 1,
                    n_jobs = -1,
                    return_train_score = True)

    grid.fit(X_train, y_train);

    output = {}
    output['Vectorizer'] = Vectorizer
    output['Classifier'] = Classifier
    output['best_params'] = grid.best_params_
    output['train_accuracy'] = grid.best_score_
    output['test_accuracy'] = grid.score(X_test, y_test)
    output['train_recall'] = recall_score(y_train, grid.predict(X_train))
    output['test_recall'] = recall_score(y_test, grid.predict(X_test))

    df = df.append(output, ignore_index=True)

    df.to_csv('./datasets/results.csv', index=False)
