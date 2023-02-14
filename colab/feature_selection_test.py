from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Perceptron, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import random as r
from pprint import pprint
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import BernoulliRBM

def to_label(y):
  return ['Anti-vaccine', 'Pro-vaccine'][int(y)]

data = pd.read_csv("a3_train_round1.tsv", sep='\t', names=['Annotation', 'Comment'], header=None)

data = data.sample(frac=1.0, random_state=0)

X_orig = data['Comment']

Y = data['Annotation'].apply(to_label)

tf_idf = TfidfVectorizer()
X = tf_idf.fit_transform(X_orig) 

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

percentile = SelectPercentile()
percentile.fit_transform(Xtrain, Ytrain)

sgd = SGDClassifier(random_state=1)

sgd.fit(Xtrain, Ytrain)

prediction = sgd.predict(Xtest)

print('Percentile value', accuracy_score(prediction, Ytest))
