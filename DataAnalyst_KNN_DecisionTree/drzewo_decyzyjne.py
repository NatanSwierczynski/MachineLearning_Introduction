import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("train.csv", sep=";")
#print(data.head())

# skupienie się na konkretnych danych w zależności od podgrupy
data = data.drop(data[(data['age']>65) | (data['age']<18)].index)
data = data.drop(data[(data['hours-per-week']>60) | (data['hours-per-week']<20)].index)
label_encoder = preprocessing.LabelEncoder()

age = list(data["age"])
workclass = label_encoder.fit_transform(list(data["workclass"]))
mapping_workclass = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
marital = label_encoder.fit_transform(list(data["marital-status"]))
mapping_marital = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
occupation = label_encoder.fit_transform(list(data["occupation"]))
mapping_occupation = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
relationship = label_encoder.fit_transform(list(data["relationship"]))
mapping_relationship = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
#race = label_encoder.fit_transform(list(data["race"]))
gender = label_encoder.fit_transform(list(data["gender"]))
mapping_gender = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
#nat_country = label_encoder.fit_transform(list(data["native-country"]))
hours_per_week = label_encoder.fit_transform(list(data["hours-per-week"]))
years_of_education = label_encoder.fit_transform(list(data["educational-num"]))
#education = label_encoder.fit_transform(list(data["education"]))
income_greater_than_50K  = label_encoder.fit_transform(list(data["income_>50K"]))

## Printowanie legendy - wyjaśnienie co oznaczają poszczególne liczby
# print(f'\nRelationship {mapping_relationship}')
# print(f'Workclass {mapping_workclass}')
# print(f'Occupation {mapping_occupation}')
# print(f'Marital {mapping_marital}')
# print(f'Gender {mapping_gender}')

#print(workclass)

predict = "income_>50K"

# Usuniecie stad jednej z kolumn powoduje nie wzięcia jej pod uwage podczas predykcji
x = list(
    zip(
        age,
        workclass,
        marital,
        occupation,
        relationship,
        # race,
        gender,
        # nat_country,
        # education,
        hours_per_week,
        years_of_education
    )
)
y = list(data["income_>50K"])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

##KNN
# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(x_train, y_train)
# accuracy = model.score(x_test, y_test)
#
# print(f"Avg Acc is {accuracy*100:.2f}%")
#
# predicted = model.predict(x_test)

## D Tree
from sklearn.tree import DecisionTreeClassifier

# Budowa modelu
model = DecisionTreeClassifier(criterion="gini",max_depth=4,min_samples_leaf=5)
# Uczenie modelu
model.fit(x_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(x_test)

print(f'\naccuracy: {accuracy_score(y_test, y_pred)}\n')
print(f'confusion matrix:\n {confusion_matrix(y_test, y_pred)}\n')
print(classification_report(y_test, y_pred))

# Wizualizacja drzewa decyzyjnego
from sklearn.tree import export_graphviz
import graphviz

columns_names = ['age','workclass','marital','occupation','relationship','gender','hours_per_week','years_of_education']
export_graphviz(model, out_file = "model.dot", filled=True, feature_names = columns_names, class_names=['over 50k', 'under 50k'])

with open("model.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


##Optymalizacja parametrów drzewa decyzyjnego
# from sklearn.model_selection import GridSearchCV
#
# params = {
#     'max_depth' : range(1,10),
#     'criterion' : ['gini', 'entropy'],
#     'min_samples_leaf' : range(2,20)
# }
#
# model_ = GridSearchCV(estimator = DecisionTreeClassifier(), cv=5, param_grid=params)
# model_.fit(x, y)
#
# print(f'Best score: {model_.best_score_}')
# #model_.best_estimator_
# model = model_.best_estimator_
# y_pred = model.predict(x_test)
#
# print(f'\naccuracy: {accuracy_score(y_test, y_pred)}\n')
# print(f'confusion matrix:\n {confusion_matrix(y_test, y_pred)}\n')
# print(classification_report(y_test, y_pred))

## Zmiana charakterystyki zbioru uczącego w zależnośći od rozmiaru zbioru uczącego
from sklearn.model_selection import learning_curve

train_sizes = range(1,3000)

features = ['age','workclass','marital','occupation','relationship','gender','hours_per_week','years_of_education']
target = 'income_>50K'

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=DecisionTreeClassifier(max_depth=4),
    X = x,
    y = y,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    shuffle=True)

print('Training scorse:\n', train_scores)
print()
print('Validation scores:\n', validation_scores)

train_scores_mean = train_scores.mean(axis=1)
validation_scores_mean = validation_scores.mean(axis=1)

print('Mean training scores\n', pd.Series(train_scores_mean, index = train_sizes))
print()
print('\nMean validation scores\n', pd.Series(validation_scores_mean, index = train_sizes))

import matplotlib.pyplot as plt
#%matplotlib inline

plt.style.use('seaborn')

plt.plot(train_sizes, 1-train_scores_mean, label = 'Training error')
plt.plot(train_sizes, 1-validation_scores_mean, label = 'Validation error')

plt.figure(1)
plt.ylabel('Error', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a decision tree model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,1)

## Wpływ parametru głębokości drzewa na accuracy
plt.figure(2)
plt.ylabel('Score', fontsize = 14)
plt.xlabel('Max_depth', fontsize = 14)
plt.title('Validation Curve for DecisionTreeClassifer', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,1)

import numpy as np
from yellowbrick.model_selection import validation_curve

from sklearn.tree import DecisionTreeClassifier

viz = validation_curve(
    DecisionTreeClassifier(), x, y, param_name="max_depth",
    param_range=np.arange(1, 25), cv=10, scoring="accuracy",
)

## Inny chyba niedziałający przykłąd
# from yellowbrick.model_selection import LearningCurve
# from sklearn.model_selection import StratifiedKFold
#
# cv = StratifiedKFold(n_splits=10)
# sizes = np. linspace(0.1, 1.0, 100)
#
# x_lc = preprocessing.OneHotEncoder().fit_transform(x)
# y_lc = preprocessing.LabelEncoder().fit_transform(y)
#
# model = DecisionTreeClassifier()
# visualizer = LearningCurve(
#     model,
#     cv=cv,
#     scoring='accuracy',
#     train_sizes=sizes,
# )
# visualizer.fit(x_lc, y_lc)
# visualizer.show()

plt.show()

