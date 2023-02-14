import random as r
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Perceptron, RidgeClassifier, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import SelectPercentile

from sklearn.model_selection import train_test_split, cross_val_score
import re


def to_label(y):
    return ['Anti-vaccine', 'Pro-vaccine'][int(y)]


data = pd.read_csv("a3_train_final.tsv", sep='\t', names=[
                   'Annotation', 'Comment'], header=None)


print(data)

to_drop = []
for i, sample in enumerate(data['Annotation']):
    if (re.match('1/0|0/1|-1', str(sample))):
        to_drop.append(i)
    else:
        if (re.match('1+', str(sample))):
            data['Annotation'][i] = 1
        else:
            data['Annotation'][i] = 0

data = data.drop(axis='index', index=to_drop)
data.drop_duplicates(subset="Comment")
print(data)

data = data.sample(frac=1.0, random_state=0).reset_index(drop=True)
print(data)
X = data['Comment']

Y = data['Annotation'].apply(to_label)

print(X)
print(Y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)


steps = [('vect', TfidfVectorizer()), ('feature_select', SelectPercentile()), ('classif', SGDClassifier(max_iter=400, n_jobs=-1))]

pipeline = Pipeline(steps)

print(pipeline.get_params().keys())

model_params = {
              'vect__ngram_range': [(1,2), (1,1)],
              'vect__max_df': [0.75, 0.8, 0.85],
              'vect__strip_accents': ['unicode'],
              'feature_select__percentile': [80, 85],
              #'feature_select__score_func': [f_classif, chi2],
              'classif__penalty': ['l2'],
             # 'classif__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
             'classif__loss': ['hinge', 'log'],
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



