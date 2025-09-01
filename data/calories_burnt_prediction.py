import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# combining two data file into DataFrames if need arise
# calories_data = pd.concat([excercise_data, calories['Calories']], axis=1)

data_set = pd.read_csv('calories.csv')
# display the filed using (calories_data.head())

gender_count = data_set['Age'].value_counts(normalize=True)*100
# print(gender_count)

sns.barplot(x=gender_count.index, y=gender_count.values)
plt.title("Gender Distribution (%)")
plt.ylabel("Percentage")
plt.xlabel("Gender")
# plt.show()

# plot graphs using seaborn (sns.set())
sns.set()
sns.countplot(x='Gender', data=data_set)
plt.title('Gender Distribution')
# plt.show()

# Age distribution
sns.displot(data_set['Age'], bins=30)
plt.title('Age Distribution')
# plt.show()


# encode the gender into numerical values (male = 0, female =1)
# data_set['Gender'] = data_set['Gender'].replace({'male': 0, 'female': 1})
data_set['Gender'] = data_set['Gender'].map({'male': 0, 'female': 1})

# print(data_set.head())

# drop user_id and calories
X = data_set.drop(['User_ID', 'Calories'], axis=1)
Y = data_set['Calories']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state=2)

model = XGBRegressor()
model.fit(X_train, Y_train)


#  Find the MAE (Mean absolute error)
# Training set MAE
train_pred = model.predict(X_train)
train_mae = metrics.mean_absolute_error(Y_train, train_pred)
print("Train MAE:", train_mae)

# Test set MAE
test_pred = model.predict(X_test)
test_mae = metrics.mean_absolute_error(Y_test, test_pred)
print("Test MAE:", test_mae)

# making prediction 
input_data = (1,20,166,60,14,94,40.3)
input_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_array)
print('Calories Burnt Prediction:', prediction)