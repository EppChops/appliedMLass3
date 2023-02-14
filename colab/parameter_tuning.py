import random as r
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Perceptron, RidgeClassifier, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectPercentile

from sklearn.model_selection import train_test_split, cross_val_score

def to_label(y):
  return ['Anti-vaccine', 'Pro-vaccine'][int(y)]

data = pd.read_csv("a3_train_round1.tsv", sep='\t', names=['Annotation', 'Comment'], header=None)

data = data.sample(frac=1.0, random_state=0)

X = data['Comment']

Y = data['Annotation'].apply(to_label)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)


steps = [('vect', TfidfVectorizer()), ('feature_select', SelectPercentile()), ('classif', SGDClassifier(n_jobs=-1))]

pipeline = Pipeline(steps)

print(pipeline.get_params().keys())

model_params = {
              'vect__ngram_range': [(1,2), (1,4)],
              'feature_select__percentile': [75, 80, 85],
              #'feature_select__score_func': ['f_classif', 'mutual_info_classif', 'chi2'],
              'classif__penalty': ['l2'],
             # 'classif__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
             'classif__loss': ['hinge', 'log_loss'],
#              'classif__eta0': [0.01, 0.001],
              'classif__alpha': np.linspace(0.1e-08, 0.001, num=10).tolist()
}


search = GridSearchCV(
  estimator=pipeline,
  param_grid=model_params, 
  cv=4,
  n_jobs=-1
)


search.fit(Xtrain, Ytrain)
pprint(search.best_estimator_.get_params())

print('Best score:', search.best_score_)

#feature_scores = f_classif(Xtrain, Ytrain)[0]
#for score, fname in sorted(zip(feature_scores, vectorizer.get_feature_names()), reverse=True)[:10]:
#    print(fname, score)


