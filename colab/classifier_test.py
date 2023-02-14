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
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import BernoulliRBM

def to_label(y):
  return ['Anti-vaccine', 'Pro-vaccine'][int(y)]

data = pd.read_csv("a3_train_round1.tsv", sep='\t', names=['Annotation', 'Comment'], header=None)

data = data.sample(frac=1.0, random_state=0)

X = data['Comment']

Y = data['Annotation'].apply(to_label)



vect = TfidfVectorizer()

X = vect.fit_transform(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

bayes = MultinomialNB()

bayes.fit(Xtrain, Ytrain)

prediction = bayes.predict(Xtest)

print('Bayes value', accuracy_score(prediction, Ytest))

sgd = SGDClassifier(random_state=1)

sgd.fit(Xtrain, Ytrain)

prediction = sgd.predict(Xtest)

print('SGD value', accuracy_score(prediction, Ytest))

knn = KNeighborsClassifier()

knn.fit(Xtrain, Ytrain)

prediction = knn.predict(Xtest)

print('KNN value', accuracy_score(prediction, Ytest))

perceptron = Perceptron(random_state=1)

perceptron.fit(Xtrain, Ytrain)

prediction = perceptron.predict(Xtest)
print('Perceptron value', accuracy_score(prediction, Ytest))




