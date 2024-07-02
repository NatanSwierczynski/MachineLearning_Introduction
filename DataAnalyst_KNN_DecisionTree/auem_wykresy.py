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
needed_to_map = "workclass,marital,occupation,relationship,gender".split(",")
# Usuniecie stad jednej z kolumn powoduje nie wzięcia jej pod uwage podczas predykcji
x = list(
    zip(
        age,
        workclass,
        marital,
        occupation,
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

model = KNeighborsClassifier(n_neighbors=31)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Average Accenture is {accuracy*100:.2f}%")
predicted = model.predict(x_test) # y testowy
names = ["Yes", "No"]

# for x in range(len(predicted)):
#     print("Predicted: ",names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

# Dopasowanie danych trenowanych i wytrenowanych
kolumny = "age;workclass;marital;occupation;relationship;gender;hours-per-week;years_of_education".split(";")
dane_tren = pd.DataFrame(x_test,y_test, columns = kolumny)
dane_predicted = pd.DataFrame(x_test,predicted, columns = kolumny)

for need, m in zip(needed_to_map, reversed_maps):
    dane_tren[need].replace(m, inplace=True)
    dane_predicted[need].replace(m, inplace=True)

print(dane_predicted.head())

print("Ammount of predicted data: ", len(dane_predicted))

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

'''Wstępna analiza danych - wykresy kolumnowe z wartościami'''
# Wstępna analiza danych - przedstawienie wykresów kolumnowych z wartościami (można pominąć)
# lab = "workclass"
# edu = "education"
# gender = "gender"
# occupation = "occupation"
# plt.figure(2)
# ax = sns.countplot(x=data[lab], order=data[lab].value_counts(ascending=False).index)
# labels = data[lab].value_counts(ascending=False).values
# ax.bar_label(container=ax.containers[0], labels=labels)
#
# plt.figure(3)
# ax = sns.countplot(x=data[edu], order=data[edu].value_counts(ascending=False).index)
# labels = data[edu].value_counts(ascending=False).values
# ax.bar_label(container=ax.containers[0], labels=labels)
#
# plt.figure(4)
# ax = sns.countplot(x=data[gender], order=data[gender].value_counts(ascending=False).index)
# labels = data[gender].value_counts(ascending=False).values
# ax.bar_label(container=ax.containers[0], labels=labels)
#
# plt.figure(5)
# ax = sns.countplot(x=data[occupation], order=data[occupation].value_counts(ascending=False).index)
# labels = data[occupation].value_counts(ascending=False).values
# ax.bar_label(container=ax.containers[0], labels=labels)


#Główny wykres wszystkich porównań (dosyć długo się ładuje)
# plt.figure(6)
# df_pair = data.drop(columns=["fnlwgt"])
# sns.pairplot(data=df_pair, hue="income_>50K")
# plt.show()

'''Wykresy danych ogólnych + subplot danych wytrenowanych i danych przewidywanych zależnie od kategorii'''

########### Marital
plt.figure(figsize=(14,6))
sns.countplot(data = data, x="marital-status", hue="income_>50K")
plt.title("General data")
plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])

fig, axes = plt.subplots(2,1, figsize = (16,10))
sns.countplot(ax = axes[0], data = dane_tren, x = "marital", hue = dane_tren.index)
for p in axes[0].patches:
    axes[0].annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
axes[0].set_title("Data trained")
axes[0].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])

sns.countplot(ax = axes[1], data = dane_predicted, x = "marital", hue = dane_predicted.index)
for p in axes[1].patches:
    axes[1].annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
axes[1].set_title("Data predicted")
axes[1].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])

# # ############ Gender
# plt.figure(2)
# sns.countplot(data=data, x = "gender", hue="income_>50K")
# plt.title("General data")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
# #
# fig, axes = plt.subplots(1,2, figsize = (14,6))
# sns.countplot(ax = axes[0], data = dane_tren, x = "gender", hue = dane_tren.index)
# for p in axes[0].patches:
#     axes[0].annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# axes[0].set_title("Data trained")
# axes[0].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# sns.countplot(ax = axes[1], data = dane_predicted, x = "gender", hue = dane_predicted.index)
# for p in axes[1].patches:
#     axes[1].annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# axes[1].set_title("Data predicted")
# axes[1].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# # ######## Age
# plt.figure(figsize=(14,6))
# sns.countplot(data = data, x="age", hue="income_>50K")
# plt.title("General data")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# fig, axes = plt.subplots(2,1, figsize = (16,10))
# sns.countplot(ax = axes[0], data = dane_tren, x = "age", hue = dane_tren.index)
# axes[0].set_title("Data trained")
# axes[0].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# sns.countplot(ax = axes[1], data = dane_predicted, x = "age", hue = dane_predicted.index)
# axes[1].set_title("Data predicted")
# axes[1].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])

# # ###### Hours per week
# plt.figure(figsize=(14,6))
# sns.countplot(data = data, x = "hours-per-week", hue="income_>50K")
# plt.title("General data")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# fig, axes = plt.subplots(2,1, figsize = (14,10))
# sns.countplot(ax = axes[0], data = dane_tren, x = "hours-per-week", hue = dane_tren.index)
# axes[0].set_title("Data trained")
# axes[0].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# sns.countplot(ax = axes[1], data = dane_predicted, x = "hours-per-week", hue = dane_predicted.index)
# axes[1].set_title("Data predicted")
# axes[1].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
# #
# # ### Workclass
# plt.figure(figsize=(14,6))
# sns.countplot(data=data, x = "workclass", hue="income_>50K")
# plt.title("General data")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# fig, axes = plt.subplots(2,1, figsize = (16,10))
# sns.countplot(ax = axes[0], data = dane_tren, x = "workclass", hue = dane_tren.index)
# for p in axes[0].patches:
#     axes[0].annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# axes[0].set_title("Data trained")
# axes[0].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# sns.countplot(ax = axes[1], data = dane_predicted, x = "workclass", hue = dane_predicted.index)
# for p in axes[1].patches:
#     axes[1].annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# axes[1].set_title("Data predicted")
# axes[1].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# ### Relationship
# plt.figure(figsize=(12,6))
# sns.countplot(data=data, x = "relationship", hue="income_>50K")
# plt.title("General data")
# plt.legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# fig, axes = plt.subplots(2,1, figsize = (14,10))
# sns.countplot(ax = axes[0], data = dane_tren, x = "relationship", hue = dane_tren.index)
# for p in axes[0].patches:
#     axes[0].annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# axes[0].set_title("Data trained")
# axes[0].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])
#
# sns.countplot(ax = axes[1], data = dane_predicted, x = "relationship", hue = dane_predicted.index)
# for p in axes[1].patches:
#     axes[1].annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha="center", va="bottom", color='black', size=10)
# axes[1].set_title("Data predicted")
# axes[1].legend(title='Income_>50K', loc='upper right', labels=['No', "Yes"])

plt.show()