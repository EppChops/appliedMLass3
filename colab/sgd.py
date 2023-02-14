from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Perceptron, RidgeClassifier, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
import random as r
from pprint import pprint
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import f_classif, SelectKBest

#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split, cross_val_score

def to_label(y):
  return ['Anti-vaccine', 'Pro-vaccine'][int(y)]

data = pd.read_csv("a3_train_round1.tsv", sep='\t', names=['Annotation', 'Comment'], header=None)

data = data.sample(frac=1.0, random_state=0)

X = data['Comment']

Y = data['Annotation'].apply(to_label)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)




#clf = RandomizedSearchCV(
#    estimator=SGDClassifier(),
#    param_distributions=model_params, 
#    n_iter=10,
#    cv=10,
#    random_state=1
#)

#model = clf.fit(Xtrain, Ytrain)
#pprint(model.best_estimator_.get_params())

#predictions = model.predict(Xtest)




#print(cross_val_score(SGDClassifier(penalty='l2', learning_rate='adaptive', eta0=0.01), Xtrain_v, Ytrain, cv=10).mean())

steps = [('vect', TfidfVectorizer()), ('feature_select', SelectKBest()), ('classif', SGDClassifier(n_jobs=-1))]

pipeline = Pipeline(steps)

print(pipeline.get_params().keys())

model_params = {
              'vect__ngram_range': [(1,2), (1,3), (1,4), (2,3), (2,4)],
              'feature_select__k': randint(1000, 8000),
              'classif__penalty': ['l1', 'l2', 'elasticnet'],
              'classif__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
              'classif__loss': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
              'classif__eta0': uniform(0.00001, 0.2),
              'classif__alpha': uniform(0.0000001, 0.1)
}


search = RandomizedSearchCV(
  estimator=pipeline,
  param_distributions=model_params, 
  n_iter=50,
  cv=10,
  random_state=1,
  n_jobs=-1
)


search.fit(Xtrain, Ytrain)
pprint(search.best_estimator_.get_params())
print('Best score', search.best_score_)

#feature_scores = f_classif(Xtrain, Ytrain)[0]
#for score, fname in sorted(zip(feature_scores, vectorizer.get_feature_names()), reverse=True)[:10]:
#    print(fname, score)


