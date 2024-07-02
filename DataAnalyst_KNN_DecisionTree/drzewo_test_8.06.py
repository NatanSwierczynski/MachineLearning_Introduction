from cProfile import label
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("train.csv", sep=";")
print(data.head())
print("Ammount of analised data: ", len(data))

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
#mapping_race = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
gender = label_encoder.fit_transform(list(data["gender"]))
mapping_gender = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
#nat_country = label_encoder.fit_transform(list(data["native-country"]))
#mapping_nat_country = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
hours_per_week = list(data["hours-per-week"])
years_of_education = list(data["educational-num"])
#education = label_encoder.fit_transform(list(data["education"]))
#mapping_education = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
income_greater_than_50K  = list(data["income_>50K"])
predict = "income_>50K"

#maps = [mapping_workclass, mapping_marital, mapping_occupation, mapping_relationship, mapping_race, mapping_gender, mapping_nat_country, mapping_education]
maps = [mapping_workclass, mapping_marital, mapping_occupation, mapping_relationship, mapping_gender]
reversed_maps = []

for m in maps:
    m = {v : k for k, v in m.items()}
    reversed_maps.append(m)

#needed_to_map = "workclass,marital,occupation,relationship,race,gender,nat_country,education".split(",")
#needed_to_map = "workclass,marital,occupation,relationship,gender".split(",")
needed_to_map = "relationship,gender".split(",")
# Usuniecie stad jednej z kolumn powoduje nie wzięcia jej pod uwage podczas predykcji
x = list(
    zip(
        age,
        #workclass,
        #marital,
        #occupation,
        relationship,
        #race,
        gender,
        #nat_country,
        #education,
        hours_per_week,
        years_of_education
    )
)
y = list(data["income_>50K"])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#print(x_train, y_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Average Accenture is {accuracy*100:.2f}%")
predicted = model.predict(x_test) # y testowy
names = ["Yes", "No"]

# for x in range(len(predicted)):
#     print("Predicted: ",names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

# Dopasowanie danych trenowanych i wytrenowanych
#kolumny = "age;workclass;marital;occupation;relationship;gender;hours-per-week;years_of_education".split(";")
kolumny = "age;relationship;gender;hours-per-week;years_of_education".split(";")
dane_tren = pd.DataFrame(x_test,y_test, columns = kolumny)
dane_predicted = pd.DataFrame(x_test,predicted, columns = kolumny)

for need, m in zip(needed_to_map, reversed_maps):
    dane_tren[need].replace(m, inplace=True)
    dane_predicted[need].replace(m, inplace=True)

print(dane_predicted.head())

print("Ammount of analised data: ", len(dane_predicted))

'''Pętla if'''
# print(dane_predicted.index.tolist())
#
# from sklearn.metrics import accuracy_score
# q = 1
# max = 0
# q_max = q
# while q < 99:
#     if q == 1:
#         model = KNeighborsClassifier(n_neighbors=1)
#         model.fit(x_train, y_train)
#         predicted = model.predict(x_test)
#         max_score = accuracy_score(y_test, predicted)
#         q += 2
#         continue
#     model_1 = KNeighborsClassifier(n_neighbors=q)
#     model_1.fit(x_train, y_train)
#     predicted1 = model_1.predict(x_test)
#     score = accuracy_score(y_test, predicted1)
#     print(q,score)
#     if score > max_score:
#         model = model_1
#         model.fit(x_train, y_train)
#         predicted = model.predict(x_test)
#         max_score = accuracy_score(y_test, predicted)
#         q_max = q
#     q += 2
#
# print(f"q maksymalne dla {q_max=}")

'''Przedstawienie predykcji - przewidywanie czy przekroczy >= 50k $ ; dane w formie liczb; aktualny stan (przekroczy lub nie)'''
# plt.figure(9)
# sns.countplot(data=data, x = "relationship", hue="income_>50K")
#
# plt.figure(1)
# ax = sns.countplot(data = dane_tren, x = "marital", hue = dane_tren.index)
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# plt.title("Data trained")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# plt.figure(2)
# ax = sns.countplot(data = dane_predicted, x = "marital", hue = dane_predicted.index)
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# plt.title("Data predicted")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# plt.figure(3)
# ax = sns.countplot(data = dane_tren, x = "gender", hue = dane_tren.index)
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# plt.title("Data trained")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# plt.figure(4)
# ax = sns.countplot(data = dane_predicted, x = "gender", hue = dane_predicted.index)
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# plt.title("Data predicted")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# plt.figure(7)
# sns.countplot(data = dane_predicted, x = "hours-per-week", hue = dane_predicted.index)
# plt.title("Data predicted")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# plt.figure(8)
# sns.countplot(data = dane_tren, x = "hours-per-week", hue = dane_tren.index)
# plt.title("Data trained")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# plt.figure(11)
# ax = sns.countplot(data = dane_tren, x = "relationship", hue = dane_tren.index)
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# plt.title("Data trained")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# plt.figure(12)
# ax = sns.countplot(data = dane_predicted, x = "relationship", hue = dane_predicted.index)
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# plt.title("Data predicted")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# plt.show()

## D Tree

from sklearn.tree import DecisionTreeClassifier

# Budowa modelu
model_D = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_leaf=5)
# Uczenie modelu
model_D.fit(x_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model_D.predict(x_test)

print(f'\naccuracy: {accuracy_score(y_test, y_pred)}\n')
print(f'confusion matrix:\n {confusion_matrix(y_test, y_pred)}\n')
print(classification_report(y_test, y_pred))

# Wizualizacja drzewa decyzyjnego
from sklearn.tree import export_graphviz
import graphviz

#columns_names = ['age','workclass','marital_status','occupation','relationship','gender','hours_per_week','years_of_education']
#bez 3 kolumn , komentowane wiersze: 208, 84, 51
columns_names = ['age','relationship','gender','hours_per_week','years_of_education']
export_graphviz(model_D, out_file = "model.dot", filled=True, feature_names = columns_names, class_names=['under 50k','over 50k'])

with open("model.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# Optymalizacja dobieranych parametrow drzewa
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
# print(model_.best_score_)
#model_.best_estimator_


## Zapisywanie drzewa do png

# from io import StringIO
# from sklearn.tree import export_graphviz
# from IPython.display import Image
# import pydotplus
#
# dot_data = StringIO()
# export_graphviz(model_D, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = columns_names, class_names=['under 50k','over 50k'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())


## Ciskowski proba
# import pydotplus
# from sklearn import tree
# from IPython.display import Image
#
# dot_data = tree.export_graphviz(model_D,
#                                 out_file=None,
#                                 feature_names=columns_names,
#                                 class_names=['under 50k','over 50k'],
#                                 filled=True,
#                                 rounded=True,
#                                 impurity=False)
# print(dot_data)