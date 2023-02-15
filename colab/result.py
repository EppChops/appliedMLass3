import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectPercentile
from operator import itemgetter

def to_label(y):
    return ['Anti-vaccine', 'Pro-vaccine'][int(y)]


training_data = pd.read_csv("a3_train_final.tsv", sep='\t', names=[
                   'Annotation', 'Comment'], header=None)


print(training_data)

to_drop = []
for i, sample in enumerate(training_data['Annotation']):
    annotations = str(sample).split('/')
    zeros = 0
    ones = 0
    negativeones = 0
    
    for annotation in annotations:
        if annotation == '0':
            zeros += 1
        elif annotation == '1':
            ones += 1
        elif annotation == '-1':
            negativeones += 1
        else:
            raise Exception("Something else slipped by the count")
    
    if zeros > ones + negativeones:
        training_data['Annotation'][i] = 0
    elif ones > zeros + negativeones:
        training_data['Annotation'][i] = 1
    else:
        to_drop.append(i)

training_data = training_data.drop(axis='index', index=to_drop)
training_data.drop_duplicates(subset="Comment")
training_data = training_data.sample(frac=1.0, random_state=0).reset_index(drop=True)

X_train = training_data['Comment']
Y_train = training_data['Annotation'].apply(to_label)


steps = [('vect', TfidfVectorizer(max_df=0.75, ngram_range=(1,2), strip_accents='unicode')), 
         #('feature_select', SelectPercentile(percentile=80)), 
         ('classif', SGDClassifier(alpha=0.00011111200000000002, max_iter=400, n_jobs=-1))]
pipeline = Pipeline(steps)

pipeline.fit(X_train, Y_train)


test_data = pd.read_csv("a3_test.tsv", sep='\t', names=[
                   'Annotation', 'Comment'], header=None)

test_data = test_data.sample(frac=1.0, random_state=0).reset_index(drop=True)
X_test = test_data['Comment']
Y_test = test_data['Annotation'].apply(to_label)

prediction = pipeline.predict(X_test)
score = accuracy_score(prediction, Y_test)
print('Final accuracy score: ', score)

names = pipeline['vect'].get_feature_names_out()
coefs = pipeline['classif'].coef_[0]

importances = zip(names, coefs)
sort = sorted(importances, key=itemgetter(1))

print('Highest feature imporance:', sort[len(sort)-10:])
print('Lowest feature importance:', sort[:10])
print('Middle feature importance', sort[len(sort)//2 - 5:len(sort)//2 + 5])

cm = confusion_matrix(Y_test, prediction)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot()
plt.show()