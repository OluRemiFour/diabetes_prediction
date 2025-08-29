# working with numpy
import numpy as np
# from sklearn.externals.array_api_compat.torch import linspace

list1  = [1,2,3,4,5]
list2 = [6,7,8,9,10]

a = np.array(list1)
b = np.array(list2)

c = np.random.randint(0,10,(3,5))
# print(np.add(a,b))

array = np.random.randint(0,10,(2,3))
# print(array)

# to find array rows and column
# print(array.shape)

# to reshape an array list
# print(array.reshape(3,2))


# ------------- Working with pandas --------------
# pandas Data frame is a two dimenssional tabular data structure with labeled axes (row and colums)
import pandas as pd

# importing the boston house price data
# from sklearn.datasets import  load_boston
# boston_dataset = load_boston()
# print(boston_dataset)

# importing data from sources:
# diabeties_df = pd.read_csv('file path')
# diabeties_df = pd.excel('file path')

# exporting data to external source:
# diabeties_df.to_csv('file name')

# creating a DataFrame with random values
# random_df = pd.DataFrame(np.random.rand(20,10))

# bring first five rows on a DataFrame
# boston_df.head()

# to get the last rows
# diabeties_df.tail()

# to get number of missing values
# diabeties = diabeties_df.isNull.sum()

# to get all statistical measures of the dataframe
# diabeties_df.describe()

# to remove a row
# diabeties_df.drop(index=0, axis=0)

# to remove a column
# diabeties_df.drop(column='column name', axis=1)

# to locate a row
# diabeties.df.illoc[2]

# --------- Working on matplotlib library ----------
import matplotlib.pyplot as plt
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# z = np.cos(x)

# print(x)
# ploting a graph
# plt.plot(x,y)

# labeling graphs
# plt.xlabel('angle')
# plt.ylabel('sin (x)')
# plt.title('sin value')
# plt.show()

# constructing a U (curve) graph
# x = np.linspace(-10,10,20)
# y = x **2
# plt.plot(x,y, 'g. ')
# plt.show()

# x = np.linspace(-5,5,50)
# plt.plot(x, np.sin(x), 'g. ')
# plt.plot(x, np.cos(x), 'r--')
# plt.show()


# ---- Quick properties ----
# Pandas: use for making a DataFrame
# Nymps: Making an arrays
# .linspace - get numbers from low to high and number of time
# .desctibe() - gives statistical measures of the data
# .value_count() - returns the number | count for each values
# .head() - print first 5 colums
# .shape - returns the number of rows and columns


# ---------- Bar chart plot ------------
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# languages = ['Yoruba', 'English', 'Egba' ,'French', 'Spanish']
# people = [100, 50, 150, 30, 20]
# ax.bar(languages, people)
# plt.xlabel('LANGUAGES')
# plt.ylabel('NO OF PEOPLE')
# plt.show()

# --------- Pie chart plot ----------
# fig1 = plt.figure()
# ax1 = fig1.add_axes([0,0,1,1])
# languages = ['Yoruba', 'English', 'Egba' ,'French', 'Spanish']
# people = [100, 50, 150, 30, 20]
# ax1.pie(people, labels=languages, autopct='%3.1f%%')
# # plt.show()


# ----- Scartter chart plot ------
# x = np.linspace(0,10,30)
# y = np.sin(x)
# z = np.cos(x)
# fig2 = plt.figure()
# ax2 = fig2.add_axes([0,0,1,1])
# ax2.scatter(x, y, c='red')
# ax2.scatter(x, z, c='blue')
# plt.show()

# ---- working with data standardlization -----
# import  seaborn as sns
# import  matplotlib.pyplot as plt


# filling missing values with Mean value
# imputation | dropping

# --- imputation ---
# dataset['fill path like: salary '].fillna(dataset['salary'].mean(),inplace = True)

# --- dropping ---
# salary_dataset = pd.read_csv('path')
# salary_dataset = salary_dataset.dropna(how='any')


# 
# 
# -------- Data Standardlization --------------
import  numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)


X = df
Y = dataset.target


# Splitting the data into training data and test data before STANDARDIZATION
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# print(X.shape, X_test.shape, X_train.shape)
# print(dataset.data.std())

# # Note:- the result of print(dataset.data.std()): shows that the dataset are not in the same range 
# it vaires alot, so we'll make then of the same range

scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train) 
# print(X_train_standardized.std())

# ------ Label Encoding ---------
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

# -------- to view the first 5 rows -----------

# Convert to DataFrame
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)

# Add diagnosis column (0 = malignant, 1 = benign)
df['diagnosis'] = cancer_data.target

# View the first 5 rows
# print(df.head())

# print(df['diagnosis'].value_counts())

# Transform the diagnosis data into 0 or 1 
# 1. load the label Encoder function 
label_encode = LabelEncoder()
labels = label_encode.fit_transform(df.diagnosis)

# append the labels to the DataFrame
df['target'] = labels

# print(df['target'].value_counts)


